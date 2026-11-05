"""NSW fleet tracking store.

Changes from the previous version:
  * CASE IDs. chat_logs and resolutions now carry a `case_id`, so a machine's chat can be
    sliced into the separate troubleshooting cases it actually contains. This is what makes
    a per-case PDF possible without guessing at "[SYSTEM] Session reopened" markers, and it
    lets a confirmed fix be linked to the diagnosis it came from.
  * Automatic, idempotent migration: an existing database is upgraded in place and every
    pre-existing row is backfilled to case_id = 0. Nothing is lost.
  * save_chat_message() already returned its rowid; get_chat_history() always returns `id`,
    and app.py now keeps that id on the in-memory message too.
  * New: next_case_id(), get_case_ids(), get_machine(), get_resolution_for_case().

Unchanged from before: absolute DB_PATH, ordering by id rather than timestamp, WAL mode,
busy timeout, foreign keys, indices, and writes that never raise into the UI.
"""
import os
import sqlite3
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("NSW_DB_PATH", os.path.join(BASE_DIR, "fleet_tracking.db"))


@contextmanager
def _connect(write: bool = False):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
        if write:
            conn.commit()
    finally:
        conn.close()


def _columns(cur, table: str) -> set:
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db():
    with _connect(write=True) as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS machines (
                serial_number TEXT PRIMARY KEY, platform TEXT NOT NULL, pump TEXT NOT NULL,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, serial_number TEXT, role TEXT NOT NULL,
                content TEXT NOT NULL, case_id INTEGER NOT NULL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (serial_number) REFERENCES machines (serial_number))''')
        cur.execute('''CREATE TABLE IF NOT EXISTS resolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, serial_number TEXT, symptom TEXT,
                fix_description TEXT, case_id INTEGER NOT NULL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (serial_number) REFERENCES machines (serial_number))''')

        # MIGRATION: older databases have no case_id. Add it and backfill to 0 so every
        # historical message belongs to "case 0" of its machine.
        for table in ("chat_logs", "resolutions"):
            if "case_id" not in _columns(cur, table):
                cur.execute(f"ALTER TABLE {table} ADD COLUMN case_id INTEGER NOT NULL DEFAULT 0")
                cur.execute(f"UPDATE {table} SET case_id = 0 WHERE case_id IS NULL")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_serial ON chat_logs (serial_number, id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_case "
                    "ON chat_logs (serial_number, case_id, id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_res_serial ON resolutions (serial_number, id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_res_case "
                    "ON resolutions (serial_number, case_id)")


# --------------------------------------------------------------------------- #
# Machines
# --------------------------------------------------------------------------- #
def add_machine(serial, platform, pump):
    try:
        with _connect(write=True) as conn:
            conn.execute("INSERT INTO machines (serial_number, platform, pump) VALUES (?, ?, ?)",
                         (serial, platform, pump))
        return True
    except sqlite3.IntegrityError:
        return False        # serial already exists


def get_all_machines():
    with _connect() as conn:
        rows = conn.execute(
            "SELECT serial_number, platform, pump FROM machines "
            "ORDER BY date_added DESC, rowid DESC").fetchall()
    return [{"serial": r[0], "platform": r[1], "pump": r[2]} for r in rows]


def get_machine(serial):
    """One machine's profile, or None. Used by the history PDF, which may be generated
    without an active chat session."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT serial_number, platform, pump, date_added FROM machines "
            "WHERE serial_number = ?", (serial,)).fetchone()
    if not row:
        return None
    return {"serial": row[0], "platform": row[1], "pump": row[2], "date_added": row[3]}


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def save_chat_message(serial, role, content, case_id: int = 0):
    """Saves a single chat message. Returns the new row id, or None on failure."""
    try:
        with _connect(write=True) as conn:
            cur = conn.execute(
                "INSERT INTO chat_logs (serial_number, role, content, case_id) "
                "VALUES (?, ?, ?, ?)", (serial, role, content or "", int(case_id)))
            return cur.lastrowid
    except sqlite3.Error:
        # A failed log write must never take the diagnostic session down with it.
        return None


def get_chat_history(serial, after_id=None, case_id=None):
    """Full history for a machine, in insertion order.

    ORDER BY id, not timestamp: CURRENT_TIMESTAMP only resolves to the second.
    `after_id` returns only messages newer than that row.
    `case_id` scopes the result to one troubleshooting case.
    """
    sql = "SELECT id, role, content, case_id, timestamp FROM chat_logs WHERE serial_number = ?"
    args = [serial]
    if after_id is not None:
        sql += " AND id > ?"
        args.append(after_id)
    if case_id is not None:
        sql += " AND case_id = ?"
        args.append(int(case_id))
    sql += " ORDER BY id ASC"
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [{"id": r[0], "role": r[1], "content": r[2], "case_id": r[3], "timestamp": r[4]}
            for r in rows]


def last_message_id(serial):
    with _connect() as conn:
        row = conn.execute("SELECT MAX(id) FROM chat_logs WHERE serial_number = ?",
                           (serial,)).fetchone()
    return row[0] if row else None


def next_case_id(serial) -> int:
    """The case_id a NEW case on this machine should use.

    Called once when a machine is opened and again on every 'Reopen'. Reading the max from
    both tables means a resolution saved against a case can never be orphaned by a later
    chat-log truncation.
    """
    with _connect() as conn:
        a = conn.execute("SELECT MAX(case_id) FROM chat_logs WHERE serial_number = ?",
                         (serial,)).fetchone()[0]
        b = conn.execute("SELECT MAX(case_id) FROM resolutions WHERE serial_number = ?",
                         (serial,)).fetchone()[0]
    return max([x for x in (a, b) if x is not None] or [-1]) + 1


def get_case_ids(serial) -> list:
    """Every case on this machine, oldest first, with its first/last message id and time."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT case_id, MIN(id), MAX(id), MIN(timestamp), MAX(timestamp), COUNT(*) "
            "FROM chat_logs WHERE serial_number = ? GROUP BY case_id ORDER BY MIN(id) ASC",
            (serial,)).fetchall()
    return [{"case_id": r[0], "first_id": r[1], "last_id": r[2],
             "started": r[3], "ended": r[4], "messages": r[5]} for r in rows]


# --------------------------------------------------------------------------- #
# Resolutions
# --------------------------------------------------------------------------- #
def save_resolution(serial, symptom, fix_description, case_id: int = 0):
    """Saves the operator's confirmed fix, linked to the case it resolved."""
    try:
        with _connect(write=True) as conn:
            conn.execute(
                "INSERT INTO resolutions (serial_number, symptom, fix_description, case_id) "
                "VALUES (?, ?, ?, ?)",
                (serial, symptom or "", fix_description or "", int(case_id)))
        return True
    except sqlite3.Error:
        return False


def get_resolutions(serial):
    """Confirmed fixes for a machine, oldest first. Fed back to the model as fleet memory."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT symptom, fix_description, timestamp, case_id FROM resolutions "
            "WHERE serial_number = ? ORDER BY id ASC", (serial,)).fetchall()
    return [{"symptom": r[0], "fix": r[1], "timestamp": r[2], "case_id": r[3]} for r in rows]


def get_resolution_for_case(serial, case_id):
    """The fix recorded against one specific case, or None. Used by the case PDF."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT symptom, fix_description, timestamp FROM resolutions "
            "WHERE serial_number = ? AND case_id = ? ORDER BY id DESC LIMIT 1",
            (serial, int(case_id))).fetchone()
    if not row:
        return None
    return {"symptom": row[0], "fix": row[1], "timestamp": row[2]}


def get_fleet_resolutions(pump, exclude_serial=None, limit: int = 60):
    """Confirmed fixes on OTHER machines that use the same pump - fleet-wide learning.

    A fix confirmed on one PistonPRO 4+ is evidence for every other PistonPRO 4+; this is
    what lets the system get smarter as more fixes are recorded across the fleet.
    """
    sql = ("SELECT r.serial_number, m.platform, r.symptom, r.fix_description, r.timestamp "
           "FROM resolutions r JOIN machines m ON m.serial_number = r.serial_number "
           "WHERE m.pump = ?")
    args = [pump]
    if exclude_serial:
        sql += " AND r.serial_number != ?"
        args.append(exclude_serial)
    sql += " ORDER BY r.id DESC LIMIT ?"
    args.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [{"serial": r[0], "platform": r[1], "symptom": r[2], "fix": r[3], "timestamp": r[4]}
            for r in rows]
