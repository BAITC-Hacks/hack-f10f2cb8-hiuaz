import io
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from core import export_docx, parse_text
from protocol_store import save_protocol, load_protocol, list_protocols, display_time


class ProtocolStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'archive.sqlite3'
        self.bundle = {'title': 'Жиналыс', 'meeting_date': None, 'summary': 'Тексеру',
                       'actions': [], 'transcript': [], 'review_status': 'draft'}

    def test_reopen_and_update_preserve_creation_date(self):
        first = datetime.fromisoformat('2026-09-23T18:01:02+05:00')
        with patch('protocol_store.datetime') as clock:
            clock.now.return_value = first
            saved = save_protocol(self.bundle, b'first document', path=self.path)
            clock.now.return_value = first + timedelta(days=1)
            unchanged = save_protocol(self.bundle, b'first document', saved['id'], self.path)
            self.assertEqual(saved, unchanged)
            changed = {**self.bundle, 'summary': 'Түзетілген мәтін'}
            updated = save_protocol(changed, b'updated document', saved['id'], self.path)
        record = load_protocol(saved['id'], self.path)
        self.assertEqual(record['bundle'], changed)
        self.assertIsNone(record['bundle']['meeting_date'])
        self.assertEqual(record['document'], b'updated document')
        self.assertEqual(updated['created_at'], saved['created_at'])
        self.assertNotEqual(updated['updated_at'], saved['updated_at'])
        self.assertEqual(display_time(saved['created_at']), '23.09.2026 18:01:02')
        self.assertEqual(len(list_protocols(self.path)), 1)

    def test_same_title_records_are_distinct_and_selectable(self):
        a = save_protocol(self.bundle, b'one', path=self.path)
        b = save_protocol(self.bundle, b'two', path=self.path)
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(len(list_protocols(self.path)), 2)
        self.assertEqual(load_protocol(a['id'], self.path)['document'], b'one')
        self.assertEqual(load_protocol(b['id'], self.path)['document'], b'two')

    def test_kazakh_docx_labels_and_archive_roundtrip(self):
        from docx import Document
        rows = parse_text('Айдана: Есепті жіберіңіз.')
        document = export_docx('Жиналыс', None, 'Тексеру', [], rows, 'Мәтін', ui_language='kk')
        saved = save_protocol(self.bundle, document, path=self.path)
        reopened = Document(io.BytesIO(load_protocol(saved['id'], self.path)['document']))
        text = '\n'.join(p.text for p in reopened.paragraphs)
        self.assertIn('Жиналыс күні: көрсетілмеген', text)
        self.assertIn('Тапсырмалар', text)
        self.assertNotIn('Дата совещания', text)
