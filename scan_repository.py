"""Scan log persistence. SQLite on Streamlit Community Cloud is NOT durable: it is lost on
restart or redeploy. The ScanRepository interface lets it be swapped for SharePoint Lists,
Azure SQL, PostgreSQL or Dataverse without touching the UI."""
import logging
import sqlite3
import time
from contextlib import closing

import pandas as pd

log = logging.getLogger('fleetscan')

COLUMNS = {
    'id': 'INTEGER PRIMARY KEY', 'scan_id': 'TEXT', 'username': 'TEXT', 'display_name': 'TEXT',
    'login_time': 'TEXT', 'scanned_on': 'TEXT', 'scanned_by': 'TEXT', 'action': 'TEXT',
    'scan_type': 'TEXT', 'scanned_value': 'TEXT', 'confirmed_vin': 'TEXT', 'confirmed_plate': 'TEXT',
    'owner': 'TEXT', 'unit': 'TEXT', 'make': 'TEXT', 'model': 'TEXT', 'year': 'TEXT',
    'branch': 'TEXT', 'location': 'TEXT', 'status': 'TEXT', 'match_method': 'TEXT',
    'match_confidence': 'REAL', 'ocr_confidence': 'REAL', 'source_file': 'TEXT',
    'photo_file_name': 'TEXT', 'photo_folder': 'TEXT', 'photo_url': 'TEXT',
    'photo_status': 'TEXT', 'notes': 'TEXT',
}


def migrate_schema(conn):
    """Create scan_log if missing and add any missing columns. Never drops data."""
    conn.execute('CREATE TABLE IF NOT EXISTS scan_log (' + ','.join(f'{k} {v}' for k, v in COLUMNS.items()) + ')')
    existing = {r[1] for r in conn.execute('PRAGMA table_info(scan_log)').fetchall()}
    for name, definition in COLUMNS.items():
        if name not in existing and name != 'id':
            conn.execute(f'ALTER TABLE scan_log ADD COLUMN {name} {definition.replace("PRIMARY KEY", "")}')
    conn.commit()


class ScanRepository:
    durable = False

    def add(self, record):
        raise NotImplementedError

    def history(self, username=None, limit=5000):
        raise NotImplementedError

    def health(self):
        raise NotImplementedError


class SQLiteScanRepository(ScanRepository):
    durable = False

    def __init__(self, path):
        self.path = path
        self.init()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            conn.execute('PRAGMA journal_mode=WAL')
        except sqlite3.DatabaseError:
            pass
        return conn

    def init(self):
        with closing(self._connect()) as conn:
            migrate_schema(conn)

    def add(self, record):
        cols = [k for k in record if k in COLUMNS and k != 'id']
        sql = f'INSERT INTO scan_log({",".join(cols)}) VALUES({",".join("?" * len(cols))})'
        for attempt in range(3):
            try:
                with closing(self._connect()) as conn:
                    migrate_schema(conn)
                    conn.execute(sql, tuple(record[c] for c in cols))
                    conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if 'locked' in str(exc).lower() and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

    def history(self, username=None, limit=5000):
        sql, params = 'SELECT * FROM scan_log', []
        if username is not None:
            sql += ' WHERE username = ?'
            params.append(username)
        sql += ' ORDER BY id DESC LIMIT ?'
        params.append(int(limit))
        with closing(self._connect()) as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def health(self):
        try:
            with closing(self._connect()) as conn:
                n = conn.execute('SELECT COUNT(*) FROM scan_log').fetchone()[0]
            return True, f'SQLite OK ({n} rows). Not durable on Streamlit Community Cloud.'
        except Exception as exc:
            log.exception('db health failed')
            return False, 'Scan log unavailable'
