import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from core import ROOT, fingerprint, parse_text, validate_extraction
from protocol_store import list_protocols, load_protocol


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name) / 'protocols.sqlite3'
        patcher = patch('protocol_store.DEFAULT_PATH', self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_text_workflow_renders_and_stale_export_is_blocked(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
        app.selectbox(key='ui_language').set_value('ru').run()
        self.assertFalse(app.exception)
        app.radio[0].set_value('Текст для проверки').run()
        next(b for b in app.button if b.label == 'Использовать этот текст').click().run()
        self.assertFalse(app.exception)
        # Inject a synthetic model result to test UI only, not inference quality.
        rows = app.session_state['meeting']['rows']
        raw = {'summary': 'Учебная встреча.', 'actions': [
            {'action': 'Отправить отчет', 'owner': 'Тимур', 'deadline_raw': 'завтра',
             'evidence_ids': ['T0001', 'T0002'], 'note': ''}]}
        result = validate_extraction(raw, rows)
        app.session_state['report'] = {'result': result,
            'fingerprint': fingerprint({'rows': rows, 'anchor': None, 'roster': ''})}
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any(x.value == '3. Проверить протокол' for x in app.subheader))
        saved = list_protocols(self.store)
        self.assertEqual(len(saved), 1)
        app.run()
        self.assertEqual(len(list_protocols(self.store)), 1)
        next(c for c in app.checkbox if c.label == 'Дата совещания известна').check().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('изменились' in w.value for w in app.warning))
        self.assertEqual(len(list_protocols(self.store)), 1)

    def test_language_switch_and_saved_protocol_survive_new_session(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
        self.assertTrue(any(x.value == 'Жиналыс' for x in app.subheader))
        rows = parse_text('Айдана: Есепті жіберіңіз.')
        app.session_state['meeting'] = {'rows': rows, 'mode': 'Текст, введенный пользователем', 'partial': False, 'version': 1}
        app.session_state['report'] = {'result': {'summary': 'Сақталатын мәтін', 'actions': []},
            'fingerprint': fingerprint({'rows': rows, 'anchor': None, 'roster': ''})}
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(list_protocols(self.store)), 1)
        record_id = list_protocols(self.store)[0]['id']
        app.selectbox(key='ui_language').set_value('ru').run()
        self.assertFalse(app.exception)
        self.assertTrue(any(x.value == '3. Проверить протокол' for x in app.subheader))
        self.assertEqual(len(list_protocols(self.store)), 1)
        self.assertEqual(load_protocol(record_id, self.store)['bundle']['summary'], 'Сақталатын мәтін')
        fresh = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
        fresh.radio(key='page').set_value('saved').run()
        self.assertFalse(fresh.exception)
        self.assertEqual(fresh.selectbox(key='saved_protocol').value, record_id)
        self.assertTrue(any('Сақталатын мәтін' in m.value for m in fresh.markdown))
        self.assertEqual(len(fresh.get('download_button')), 2)
        fresh.radio(key='page').set_value('new').run()
        next(b for b in fresh.button if b.label == 'Жаңа жиналыс бастау').click().run()
        self.assertFalse(fresh.exception)
        self.assertEqual(fresh.selectbox(key='ui_language').value, 'kk')
        self.assertEqual(len(list_protocols(self.store)), 1)


if __name__ == '__main__':
    unittest.main()
