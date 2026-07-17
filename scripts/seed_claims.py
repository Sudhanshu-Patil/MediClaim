"""Seed the synthetic claims SQLite (data/claims.db) for the MCP tools.

    python scripts/seed_claims.py
"""

from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools import CLAIMS_DB  # noqa: E402

PROCEDURES = [
    ("OP-1001", 45.00), ("OP-1002", 90.00), ("OP-1003", 35.00),
    ("OP-3003", 680.00), ("OP-4001", 75.00), ("OP-5004", 720.00),
    ("OP-6001", 18.00), ("OP-8001", 95.00), ("OP-9001", 70.00),
    ("OP-9004", 310.00),
]
STATUSES = ["paid", "paid", "paid", "denied", "pending", "pending"]


def main() -> None:
    CLAIMS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CLAIMS_DB)
    conn.executescript("""
        DROP TABLE IF EXISTS claims;
        DROP TABLE IF EXISTS fraud_flags;
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            member_id TEXT NOT NULL,
            procedure_code TEXT NOT NULL,
            billed_amount REAL NOT NULL,
            allowed_amount REAL NOT NULL,
            status TEXT NOT NULL,
            service_date TEXT NOT NULL,
            provider_npi TEXT NOT NULL
        );
        CREATE TABLE fraud_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT NOT NULL REFERENCES claims(claim_id),
            reason TEXT NOT NULL,
            flagged_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_claims_member ON claims(member_id);
        CREATE INDEX idx_claims_proc ON claims(procedure_code);
    """)
    rng = random.Random(7)
    rows = []
    for i in range(1, 41):
        code, allowed = rng.choice(PROCEDURES)
        billed = round(allowed * rng.uniform(0.9, 1.8), 2)
        rows.append((
            f"CLM-{10000 + i}",
            f"MBR-{rng.randint(100000, 999999)}",
            code,
            billed,
            allowed,
            rng.choice(STATUSES),
            (date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))).isoformat(),
            f"{rng.randint(1000000000, 1999999999)}",
        ))
    conn.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"Seeded {len(rows)} claims into {CLAIMS_DB}")


if __name__ == "__main__":
    main()
