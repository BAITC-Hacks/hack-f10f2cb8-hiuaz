"""Local, transactional storage for generated protocols (no audio)."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

DEFAULT_PATH = Path(__file__).resolve().parent / 'data' / 'protocols.sqlite3'


@contextmanager
def database(path=None):
    path = Path(path) if path is not None else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        path.chmod(0o600)
        with connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS protocols (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                title TEXT NOT NULL, payload TEXT NOT NULL, document BLOB NOT NULL
            )''')
            yield connection
    finally:
        connection.close()


def save_protocol(bundle, document, record_id=None, path=None):
    record_id = record_id or uuid4().hex
    now = datetime.now(ZoneInfo('Asia/Almaty')).isoformat(timespec='seconds')
    payload = json.dumps(bundle, ensure_ascii=False, sort_keys=True)
    with database(path) as connection:
        previous = connection.execute('SELECT * FROM protocols WHERE id = ?', (record_id,)).fetchone()
        if previous is None:
            connection.execute('INSERT INTO protocols VALUES (?, ?, ?, ?, ?, ?)',
                               (record_id, now, now, bundle['title'], payload, document))
        elif previous['payload'] != payload:
            connection.execute('UPDATE protocols SET updated_at = ?, title = ?, payload = ?, document = ? WHERE id = ?',
                               (now, bundle['title'], payload, document, record_id))
        row = connection.execute('SELECT id, created_at, updated_at FROM protocols WHERE id = ?', (record_id,)).fetchone()
        return dict(row)


def list_protocols(path=None):
    with database(path) as connection:
        return [dict(row) for row in connection.execute(
            'SELECT id, title, created_at, updated_at FROM protocols ORDER BY created_at DESC, rowid DESC')]


def load_protocol(record_id, path=None):
    with database(path) as connection:
        row = connection.execute('SELECT * FROM protocols WHERE id = ?', (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        record = dict(row)
        record['bundle'] = json.loads(record.pop('payload'))
        return record


def display_time(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo('Asia/Almaty')).strftime('%d.%m.%Y %H:%M:%S')
