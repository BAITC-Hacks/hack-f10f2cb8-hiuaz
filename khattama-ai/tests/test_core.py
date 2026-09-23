import io
import json
import unittest
from core import align_words, normalize_deadline, parse_text, validate_extraction, export_docx


class DeadlineTests(unittest.TestCase):
    def test_unknown_meeting_date_is_not_today(self):
        self.assertIsNone(normalize_deadline('завтра')[0])
        self.assertIsNone(normalize_deadline('15 октября')[0])
        self.assertIsNone(normalize_deadline(None, '2026-09-23')[0])

    def test_relative_deadlines_and_ambiguous_weekday(self):
        self.assertEqual(normalize_deadline('ертең', '2026-09-23')[0], '2026-09-24')
        self.assertEqual(normalize_deadline('за две недели', '2026-09-23')[0], '2026-10-07')
        self.assertIsNone(normalize_deadline('к пятнице', '2026-09-23')[0])
        self.assertIsNone(normalize_deadline('после совещания', '2026-09-23')[0])

    def test_exact_date_and_invalid_date(self):
        self.assertEqual(normalize_deadline('до 25 сентября 2026 года')[0], '2026-09-25')
        self.assertIsNone(normalize_deadline('31.02.2026')[0])


class AttributionTests(unittest.TestCase):
    def test_literal_deadline_recovery_is_conservative_and_marked(self):
        rows = parse_text('Тимур: Әлия, келісімшартты бүгін тексеріңіз.')
        item = {'action': 'Келісімшартты тексеру', 'owner': 'Әлия', 'deadline_raw': None,
                'evidence_ids': ['T0001'], 'note': ''}
        result = validate_extraction({'summary': '', 'actions': [item]}, rows, '2026-09-23')
        self.assertEqual(result['actions'][0]['deadline_raw'], 'бүгін')
        self.assertEqual(result['actions'][0]['due_date'], '2026-09-23')
        self.assertIn('Срок взят из реплики', result['actions'][0]['note'])
        unknown = validate_extraction({'summary': '', 'actions': [item]}, rows)
        self.assertEqual(unknown['actions'][0]['due_date'], '')
        multiple = validate_extraction({'summary': '', 'actions': [item, item]}, rows, '2026-09-23')
        self.assertEqual(multiple['actions'][0]['deadline_raw'], '')
        conflicting = parse_text('Тимур: Әлия, бүгін немесе ертең келісімшартты тексеріңіз.')
        result = validate_extraction({'summary': '', 'actions': [item]}, conflicting, '2026-09-23')
        self.assertEqual(result['actions'][0]['deadline_raw'], '')

    def test_numpy_asr_timestamps_can_be_exported_as_json(self):
        import numpy as np
        words = [{'start': np.float32(1), 'end': np.float32(2), 'text': ' Проверка'},
                 {'start': np.float32(2), 'end': np.float32(3), 'text': ' экспорта'}]
        turns = [{'start': 0, 'end': 4, 'speaker_id': 'A'},
                 {'start': 0.9, 'end': 2.5, 'speaker_id': 'B'}]
        rows = align_words(words, turns)
        exported = json.loads(json.dumps(rows))
        self.assertEqual(len(exported), 1)
        self.assertIs(exported[0]['speaker_review'], True)
        self.assertEqual(exported[0]['end'], 3)

    def test_word_speaker_overlap(self):
        words = [{'start': 1, 'end': 2, 'text': ' Задача'}, {'start': 3, 'end': 4, 'text': ' выполнена'}]
        turns = [{'start': 0, 'end': 2.5, 'speaker_id': 'A'}, {'start': 0.9, 'end': 1.6, 'speaker_id': 'B'}]
        rows = align_words(words, turns)
        self.assertEqual(rows[0]['speaker_id'], 'A')
        self.assertTrue(rows[0]['speaker_review'])
        self.assertEqual(rows[1]['speaker_id'], 'UNKNOWN')

    def test_absent_assignee_and_fake_evidence(self):
        rows = parse_text('Руководитель: Ерлан должен проверить договор.\nАйдана: Передам ему.')
        raw = {'summary': 'Договор будет проверен.', 'actions': [
            {'action': 'Проверить договор', 'owner': 'Ерлан', 'deadline_raw': None, 'evidence_ids': ['T0001', 'FAKE'], 'note': ''},
            {'action': 'Уточнить детали', 'owner': 'SPEAKER_01', 'deadline_raw': 'завтра', 'evidence_ids': ['FAKE'], 'note': ''}]}
        result = validate_extraction(raw, rows)
        self.assertEqual(result['actions'][0]['owner'], 'Ерлан')
        self.assertEqual(result['actions'][0]['evidence_ids'], ['T0001'])
        self.assertEqual(result['actions'][1]['owner'], '')
        self.assertEqual(result['actions'][1]['due_date'], '')
        self.assertIn('Источник не подтвержден', result['actions'][1]['note'])

    def test_docx_preserves_ambiguity_and_source(self):
        from docx import Document
        rows = parse_text('Айдана: Сравните предложения, срок уточним.')
        action = {'id': 'A001', 'action': 'Сравнить предложения', 'owner': '', 'deadline_raw': '',
                  'due_date': '', 'evidence_ids': ['T0001'], 'note': 'Нужен исполнитель', 'verified': False}
        payload = export_docx('Проверка', None, 'Обсудили закупку.', [action], rows, 'Текст', True)
        doc = Document(io.BytesIO(payload))
        self.assertEqual(len(doc.tables[0].rows), 2)
        self.assertEqual(doc.tables[0].cell(1, 1).text, 'Не указан')
        self.assertIn('T0001', doc.tables[0].cell(1, 3).text)
        self.assertTrue(any('фрагмент' in p.text for p in doc.paragraphs))


if __name__ == '__main__':
    unittest.main()
