import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import server


class WebApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for target, value in [('protocol_store.DEFAULT_PATH', Path(self.temp.name) / 'protocols.sqlite3'),
                              ('server.JOBS', server.Jobs())]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.http = server.make_server(0)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)

    def request(self, path, data=None, headers=None, raw=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.http.server_address[1], timeout=10)
        self.addCleanup(connection.close)
        request_headers = {'X-Khattama-Request': '1', 'Content-Type': 'application/json'}
        request_headers.update(headers or {})
        body = json.dumps(data).encode() if data is not None else raw
        connection.request('POST' if body is not None else 'GET', path, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = response.read()
        if response.headers.get('Content-Type', '').startswith('application/json'):
            payload = json.loads(payload)
        return response.status, payload, response.headers

    def wait_job(self, job_id):
        for _ in range(200):
            status, job, _ = self.request('/api/jobs/' + job_id)
            self.assertEqual(status, 200)
            if job['state'] != 'running':
                return job
            time.sleep(.01)
        self.fail('Worker did not finish')

    def create_protocol(self):
        status, transcript, _ = self.request('/api/transcript', {
            'title': 'Синтетическая проверка', 'text': 'Айдана: Отправь отчет завтра.', 'ui_language': 'kk'})
        self.assertEqual(status, 200)
        extraction = {'summary': 'Учебное поручение.', 'actions': [
            {'id': 'A001', 'action': 'Отправить отчет', 'owner': 'Айдана', 'due_date': '',
             'deadline_raw': 'завтра', 'evidence_ids': ['T0001'], 'note': '', 'verified': False}]}
        with patch('server.model_status', return_value={'llm': True}), patch('server.extract_actions', return_value=extraction):
            status, started, _ = self.request('/api/jobs/report', {
                'title': 'Синтетическая проверка', 'rows': transcript['rows'], 'ui_language': 'kk'})
            self.assertEqual(status, 202)
            job = self.wait_job(started['id'])
        self.assertEqual(job['state'], 'done')
        return job['result']['protocol_id']

    def test_text_review_generation_archive_and_docx(self):
        rid = self.create_protocol()
        status, detail, _ = self.request('/api/protocols/' + rid)
        self.assertEqual(status, 200)
        self.assertIsNone(detail['bundle']['meeting_date'])
        self.assertEqual(detail['bundle']['document_language'], 'kk')
        status, doc, headers = self.request('/api/protocols/' + rid + '/docx')
        self.assertEqual(status, 200)
        self.assertTrue(doc.startswith(b'PK'))
        self.assertIn('attachment', headers['Content-Disposition'])
        _, data, _ = self.request('/api/dashboard')
        self.assertEqual(data['stats']['protocols'], 1)
        self.assertEqual(data['stats']['tasks'], 1)
        self.assertEqual(data['tasks'][0]['status'], 'pending')

    def test_edit_status_and_concurrent_update_conflict(self):
        rid = self.create_protocol()
        _, detail, _ = self.request('/api/protocols/' + rid)
        status, changed, _ = self.request('/api/protocols/' + rid, {
            'version': detail['version'], 'task_id': 'A001', 'status': 'done'})
        self.assertEqual(status, 200)
        self.assertEqual(changed['created_at'], detail['created_at'])
        status, failure, _ = self.request('/api/protocols/' + rid, {
            'version': detail['version'], 'task_id': 'A001', 'status': 'pending'})
        self.assertEqual(status, 409)
        self.assertEqual(failure['error'], 'conflict')
        _, dashboard, _ = self.request('/api/dashboard')
        self.assertEqual(dashboard['stats']['done'], 1)
        self.assertEqual(dashboard['stats']['overdue'], 0)
        status, reviewed, _ = self.request('/api/protocols/' + rid, {
            'version': changed['version'], 'title': 'Проверено', 'summary': 'Проверенный текст',
            'actions': changed['bundle']['actions'], 'reviewed': True})
        self.assertEqual(status, 200)
        self.assertTrue(reviewed['bundle']['actions'][0]['verified'])
        self.assertEqual(reviewed['bundle']['review_status'], 'reviewed')

    def test_invalid_date_does_not_overwrite_saved_record(self):
        rid = self.create_protocol()
        _, detail, _ = self.request('/api/protocols/' + rid)
        actions = detail['bundle']['actions']
        actions[0]['due_date'] = '2026-02-31'
        status, failure, _ = self.request('/api/protocols/' + rid, {
            'version': detail['version'], 'title': 'Changed', 'summary': 'Changed', 'actions': actions})
        self.assertEqual(status, 400)
        self.assertEqual(failure['error'], 'invalid_date')
        _, after, _ = self.request('/api/protocols/' + rid)
        self.assertEqual(after['bundle']['title'], 'Синтетическая проверка')

    def test_browser_origin_host_and_static_path_protection(self):
        status, _, _ = self.request('/api/dashboard', headers={'Origin': 'https://unrelated.example'})
        self.assertEqual(status, 403)
        status, _, _ = self.request('/api/dashboard', headers={'Host': 'attacker.example'})
        self.assertEqual(status, 403)
        status, _, _ = self.request('/api/transcript', {'title': 'Test', 'text': 'Hello'}, {'X-Khattama-Request': ''})
        self.assertEqual(status, 403)
        status, _, _ = self.request('/../data/protocols.sqlite3')
        self.assertEqual(status, 404)
        status, page, headers = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'name="viewport"', page)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])

    def test_only_one_model_job_runs_at_a_time(self):
        finish = threading.Event()
        job_id = server.JOBS.submit(lambda job_id: finish.wait(2))
        try:
            with self.assertRaises(server.InvalidRequest) as caught:
                server.JOBS.submit(lambda job_id: None)
            self.assertEqual(caught.exception.code, 'busy')
        finally:
            finish.set()
        self.assertEqual(self.wait_job(job_id)['state'], 'done')
