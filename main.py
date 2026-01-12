from fastmcp import FastMCP
import psycopg2
import asyncpg
from datetime import datetime
import json
import re
import os
from dotenv import load_dotenv

# LangChain + HuggingFace LLM setup
import langchain_huggingface
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from model_creato import create_model
load_dotenv()
model=create_model()

# -----------------------------
# Initialize MCP
# -----------------------------
mcp = FastMCP(name="VitalityMCP")

# -----------------------------
# Database setup
# -----------------------------
def initialise_db():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )
    cur = conn.cursor()

    # Create schema
    cur.execute("CREATE SCHEMA IF NOT EXISTS health;")
    
    # Table for daily health inputs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS health.health_inputs (
        id SERIAL PRIMARY KEY,
        sleep_hours NUMERIC(3,1) NOT NULL,
        breaks_taken INTEGER NOT NULL,
        steps_taken INTEGER,
        self_reported_fatigue INTEGER NOT NULL CHECK (self_reported_fatigue BETWEEN 1 AND 10),
        date DATE NOT NULL
    );
    """)
    
    # Table for health assessments/signals
    cur.execute("""
    CREATE TABLE IF NOT EXISTS health.health_assessments (
        id SERIAL PRIMARY KEY,
        input_id INTEGER NOT NULL REFERENCES health.health_inputs(id) ON DELETE CASCADE,
        state VARCHAR(50) NOT NULL,
        recommended_action VARCHAR(100) NOT NULL,
        confidence NUMERIC(3,2) CHECK (confidence BETWEEN 0 AND 1),
        date DATE NOT NULL
    );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# -----------------------------
# Tool: Add health data
# -----------------------------
@mcp.tool
async def add_health_data(
    sleep_hours: float,
    breaks_taken: int,
    steps_taken: int,
    self_reported_fatigue: int,
    date: str
):
    """
    Add daily health data to health_inputs table.
    """
    # Ensure date is a date object
    date = datetime.strptime(date.strip(), "%Y-%m-%d").date()

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )

    await conn.execute(
        """
        INSERT INTO health.health_inputs
        (sleep_hours, breaks_taken, steps_taken, self_reported_fatigue, date)
        VALUES ($1,$2,$3,$4,$5)
        """,
        sleep_hours, breaks_taken, steps_taken, self_reported_fatigue, date
    )

    await conn.close()
    return {"message": "Health data added successfully."}

# -----------------------------
# Tool: Generate daily health signal
# -----------------------------
@mcp.tool
async def health_signal(date: str):
    """
    Generate daily health signal for a given date using AI.
    """
    # Parse date
    date_obj = datetime.strptime(date.strip(), "%Y-%m-%d").date()

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )

    # Fetch today's health data
    row = await conn.fetchrow(
        """
        SELECT *
        FROM health.health_inputs
        WHERE date = $1
        """,
        date_obj
    )

    await conn.close()

    if not row:
        return {"date": str(date_obj), "signal": "no_data"}

    data = dict(row)

    prompt = f"""
You are a health & wellbeing analyzer.

Based on the daily input metrics, return ONLY a JSON object
with keys "state", "action", "confidence".

Allowed states:
- energized → user has enough sleep, breaks, and low fatigue
- tired → user has low sleep or high fatigue
- needs_rest → user is very fatigued or physically exhausted
- balanced → moderate sleep, breaks, and activity

Allowed actions:
- take_rest → encourage user to rest or nap
- exercise → suggest light activity or walking
- maintain → keep current routine
- increase_breaks → take more breaks if fatigued

Confidence:
- number between 0 and 1 signifying AI's confidence

Input metrics:
{data}
"""
    response = await model.ainvoke(prompt)
    text = response.content
    clean = re.sub(r"```json|```", "", text).strip()
    output = json.loads(clean)

    # Insert AI output into assessments table
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )

    await conn.execute(
        """
        INSERT INTO health.health_assessments
        (input_id, state, recommended_action, confidence, date)
        VALUES ($1,$2,$3,$4,$5)
        """,
        data['id'],
        output["state"],
        output["action"],
        output["confidence"],
        date_obj
    )

    await conn.close()

    return {"date": str(date_obj), "signal": output}

# -----------------------------
# Run MCP
# -----------------------------


if __name__ == "__main__":
    initialise_db()
    mcp.run(transport='http',host='0.0.0.0',port=8002)
