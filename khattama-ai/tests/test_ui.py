import unittest
from streamlit.testing.v1 import AppTest
from core import ROOT, fingerprint, parse_text, validate_extraction


class InterfaceTests(unittest.TestCase):
    def test_text_workflow_renders_and_stale_export_is_blocked(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
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
        next(c for c in app.checkbox if c.label == 'Дата совещания известна').check().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('изменились' in w.value for w in app.warning))


if __name__ == '__main__':
    unittest.main()
