"""Local web application. No external services, fonts or inference APIs."""
import argparse
import copy
import json
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

from core import (align_words, decode_audio, diarize, export_docx, extract_actions,
                  model_status, parse_text, transcribe, wav_bytes)
from protocol_store import list_protocols, load_protocol, save_protocol
from i18n import translate

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 65 * 1024 * 1024
MUTATION_LOCK = threading.Lock()


class InvalidRequest(Exception):
    def __init__(self, code='invalid', status=400):
        self.code, self.status = code, status


def clean_metadata(data):
    title = str(data.get('title', '')).strip()
    if not title or len(title) > 200:
        raise InvalidRequest('title_required')
    anchor = data.get('meeting_date') or None
    if anchor:
        try:
            date.fromisoformat(anchor)
        except (ValueError, TypeError):
            raise InvalidRequest('invalid_date')
    language = data.get('language', 'ru')
    ui_language = data.get('ui_language', 'kk')
    if language not in ('ru', 'kk', 'mixed') or ui_language not in ('kk', 'ru'):
        raise InvalidRequest()
    return {'title': title, 'meeting_date': anchor, 'language': language,
            'document_language': ui_language, 'roster': str(data.get('roster', ''))[:5000]}


def clean_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise InvalidRequest('empty_text')
    output, used = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InvalidRequest()
        text = str(row.get('text', '')).strip()
        if not text:
            continue
        if len(text) > 10000:
            raise InvalidRequest('text_length')
        rid = str(row.get('id', 'T%04d' % (index + 1)))
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', rid) or rid in used:
            raise InvalidRequest()
        used.add(rid)
        start, end = row.get('start'), row.get('end')
        if start is not None:
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= start <= end <= 600:
                raise InvalidRequest()
        else:
            end = None
        output.append({'id': rid, 'text': text, 'speaker': str(row.get('speaker', ''))[:200],
                       'speaker_id': str(row.get('speaker_id', 'UNKNOWN'))[:200],
                       'start': start, 'end': end, 'speaker_review': bool(row.get('speaker_review'))})
    if not output or sum(len(row['text']) for row in output) > 80000:
        raise InvalidRequest('text_length')
    return output


def document(bundle):
    language = bundle.get('document_language', 'kk')
    return export_docx(bundle['title'], bundle.get('meeting_date'), bundle.get('summary', ''),
                       bundle.get('actions', []), bundle.get('transcript', []),
                       translate(bundle.get('source', ''), language), bundle.get('partial_audio', False),
                       ui_language=language)


def public_record(record):
    return {key: value for key, value in record.items() if key != 'document'}


def dashboard():
    records, tasks = [], []
    today = datetime.now(ZoneInfo('Asia/Almaty')).date()
    for item in list_protocols():
        record = load_protocol(item['id'])
        bundle = record['bundle']
        actions = bundle.get('actions', [])
        records.append({**item, 'meeting_date': bundle.get('meeting_date'),
                        'summary': bundle.get('summary', ''), 'action_count': len(actions),
                        'review_status': bundle.get('review_status', 'draft'),
                        'partial_audio': bundle.get('partial_audio', False),
                        'source': bundle.get('source', '')})
        for action in actions:
            status = action.get('status', 'pending')
            due = None
            try:
                due = date.fromisoformat(action.get('due_date') or '')
            except ValueError:
                pass
            overdue = status != 'done' and due is not None and due < today
            upcoming = status != 'done' and due is not None and today <= due <= today + timedelta(days=3)
            tasks.append({**action, 'status': status, 'overdue': overdue, 'upcoming': upcoming,
                          'protocol_id': item['id'], 'protocol_title': bundle['title']})
    return {'protocols': records, 'tasks': tasks, 'today': today.isoformat(),
            'stats': {'protocols': len(records), 'tasks': len(tasks),
                      'done': sum(t['status'] == 'done' for t in tasks),
                      'overdue': sum(t['overdue'] for t in tasks)}}


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = None

    def submit(self, work):
        with self.lock:
            if self.active:
                raise InvalidRequest('busy', 409)
            # Completed jobs expire; only temporary audio is held here, never on disk.
            now = time.time()
            self.jobs = {key: job for key, job in self.jobs.items() if now - job['created'] < 3600}
            while len(self.jobs) >= 4:
                del self.jobs[next(iter(self.jobs))]
            job_id = uuid4().hex
            self.jobs[job_id] = {'id': job_id, 'created': now, 'state': 'running', 'phase': 'reading'}
            self.active = job_id

        def run():
            try:
                work(job_id)
                self.update(job_id, state='done')
            except InvalidRequest as exc:
                self.update(job_id, state='error', error=exc.code)
            except ValueError:
                logging.exception('Local model could not process input')
                self.update(job_id, state='error', error='processing_failed')
            except Exception:
                logging.exception('Local processing failed')
                self.update(job_id, state='error', error='processing_failed')
            finally:
                with self.lock:
                    self.active = None
        threading.Thread(target=run, daemon=True).start()
        return job_id

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def get(self, job_id, include_audio=False):
        with self.lock:
            if job_id not in self.jobs:
                raise InvalidRequest('job_expired', 404)
            return copy.deepcopy({key: value for key, value in self.jobs[job_id].items()
                                  if include_audio or key != 'audio'})


JOBS = Jobs()


def start_audio(raw, metadata, seconds=60, speakers=0):
    if seconds not in (60, 600) or not 0 <= speakers <= 12:
        raise InvalidRequest()
    if not all(list(model_status().values())[:3]):
        raise InvalidRequest('models_missing', 503)
    def work(job_id):
        samples = decode_audio(raw)
        total = len(samples) / 16000
        samples = samples[:seconds * 16000]
        JOBS.update(job_id, phase='transcribing')
        words, languages = transcribe(samples, language=metadata['language'])
        JOBS.update(job_id, phase='speakers')
        turns = diarize(samples, speakers)
        rows = align_words(words, turns)
        JOBS.update(job_id, result={'rows': rows, 'metadata': metadata, 'partial': total > seconds,
                                   'duration': len(samples) / 16000, 'languages': languages,
                                   'source': 'Локальное распознавание аудио'}, audio=wav_bytes(samples))
    return JOBS.submit(work)


def start_report(data):
    if not list(model_status().values())[-1]:
        raise InvalidRequest('models_missing', 503)
    metadata = clean_metadata(data)
    rows = clean_rows(data.get('rows'))
    source = 'Текст, введенный пользователем'
    partial = False
    if data.get('audio_job'):
        original = JOBS.get(data['audio_job'])
        if original['state'] != 'done' or 'result' not in original:
            raise InvalidRequest()
        source = original['result']['source']
        partial = original['result']['partial']
    def work(job_id):
        JOBS.update(job_id, phase='summarizing')
        result = extract_actions(rows, metadata['meeting_date'], metadata['roster'],
                                 output_language=metadata['document_language'])
        bundle = {**metadata, 'source': source, 'partial_audio': partial, 'transcript': rows,
                  'summary': result['summary'], 'actions': result['actions'], 'review_status': 'draft'}
        JOBS.update(job_id, phase='saving')
        saved = save_protocol(bundle, document(bundle))
        JOBS.update(job_id, result={'protocol_id': saved['id']})
    return JOBS.submit(work)


def update_protocol(record_id, data):
    with MUTATION_LOCK:
        record = load_protocol(record_id)
        # Optimistic concurrency prevents one browser silently overwriting another.
        from core import fingerprint
        if data.get('version') != fingerprint(record['bundle']):
            raise InvalidRequest('conflict', 409)
        bundle = copy.deepcopy(record['bundle'])
        if 'task_id' in data:
            if data.get('status') not in ('pending', 'in_progress', 'done'):
                raise InvalidRequest()
            task = next((a for a in bundle['actions'] if a['id'] == data['task_id']), None)
            if task is None:
                raise InvalidRequest('not_found', 404)
            task['status'] = data['status']
        else:
            title = str(data.get('title', '')).strip()
            if not title or len(title) > 200:
                raise InvalidRequest('title_required')
            summary = str(data.get('summary', ''))[:20000]
            actions = data.get('actions')
            if not isinstance(actions, list) or len(actions) > 200:
                raise InvalidRequest()
            cleaned, valid_ids = [], {r['id'] for r in bundle['transcript']}
            used_ids = set()
            for action in actions:
                content = str(action.get('action', '')).strip()
                if not content:
                    continue
                due = str(action.get('due_date') or '')
                if due:
                    try:
                        date.fromisoformat(due)
                    except ValueError:
                        raise InvalidRequest('invalid_date')
                aid = str(action.get('id') or uuid4().hex)
                if aid in used_ids:
                    raise InvalidRequest()
                used_ids.add(aid)
                status = action.get('status', 'pending')
                if status not in ('pending', 'in_progress', 'done'):
                    raise InvalidRequest()
                cleaned.append({'id': aid, 'action': content[:5000], 'owner': str(action.get('owner', ''))[:200],
                                'due_date': due, 'deadline_raw': str(action.get('deadline_raw', ''))[:200],
                                'note': str(action.get('note', ''))[:2000], 'status': status,
                                'verified': bool(data.get('reviewed')),
                                'evidence_ids': [rid for rid in action.get('evidence_ids', []) if rid in valid_ids]})
            bundle.update(title=title, summary=summary, actions=cleaned,
                          review_status='reviewed' if data.get('reviewed') else 'draft')
        save_protocol(bundle, document(bundle), record_id=record_id)
        return record_detail(record_id)


def record_detail(record_id):
    from core import fingerprint
    result = public_record(load_protocol(record_id))
    result['version'] = fingerprint(result['bundle'])
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = 'Khattama'

    def log_message(self, fmt, *args):
        # Never log request paths, uploaded names or meeting content.
        pass

    def respond(self, payload, status=200, content_type='application/json; charset=utf-8', filename=None):
        data = json.dumps(payload, ensure_ascii=False).encode() if not isinstance(payload, bytes) else payload
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header('Content-Disposition', 'attachment; filename="%s"' % filename)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def guard(self, mutation=False):
        host = self.headers.get('Host', '')
        if host not in self.server.allowed_hosts:
            raise InvalidRequest('forbidden', 403)
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + host:
            raise InvalidRequest('forbidden', 403)
        if mutation and self.headers.get('X-Khattama-Request') != '1':
            raise InvalidRequest('forbidden', 403)

    def body(self, limit=1024 * 1024):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            raise InvalidRequest()
        if length <= 0 or length > limit:
            raise InvalidRequest('file_size', 413)
        self.connection.settimeout(60)
        body = self.rfile.read(length)
        if len(body) != length:
            raise InvalidRequest()
        return body

    def handle_error(self, exc):
        if isinstance(exc, InvalidRequest):
            self.respond({'error': exc.code}, exc.status)
        elif isinstance(exc, KeyError):
            self.respond({'error': 'not_found'}, 404)
        elif isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError)):
            self.respond({'error': 'invalid'}, 400)
        else:
            logging.exception('Request failed')
            self.respond({'error': 'server_error'}, 500)

    def do_GET(self):
        try:
            self.guard()
            path = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
                      '/favicon.svg': ('favicon.svg', 'image/svg+xml')}
            if path in assets:
                name, mime = assets[path]
                return self.respond((ROOT / 'web' / name).read_bytes(), content_type=mime)
            if path in ('/api/health', '/_stcore/health'):
                return self.respond({'ok': True, 'models_ready': all(model_status().values())})
            if path == '/api/dashboard':
                return self.respond(dashboard())
            match = re.fullmatch(r'/api/protocols/([a-f0-9]{32})(?:/(docx|json))?', path)
            if match:
                rid, fmt = match.groups()
                if fmt == 'docx':
                    return self.respond(load_protocol(rid)['document'], content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', filename='protocol-' + rid[:8] + '.docx')
                result = record_detail(rid)
                return self.respond(result, filename=('protocol-' + rid[:8] + '.json') if fmt else None)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})(/audio)?', path)
            if match:
                job_id, audio = match.groups()
                job = JOBS.get(job_id, include_audio=bool(audio))
                if audio:
                    if 'audio' not in job:
                        raise InvalidRequest('not_found', 404)
                    return self.respond(job['audio'], content_type='audio/wav')
                return self.respond(job)
            raise InvalidRequest('not_found', 404)
        except Exception as exc:
            self.handle_error(exc)

    def do_POST(self):
        try:
            self.guard(mutation=True)
            path = urlsplit(self.path).path
            if path == '/api/jobs/audio':
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data;'):
                    raise InvalidRequest()
                message = BytesParser(policy=policy.default).parsebytes(
                    ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + self.body(MAX_UPLOAD))
                if not message.is_multipart():
                    raise InvalidRequest()
                parts = {part.get_param('name', header='content-disposition'): part for part in message.iter_parts()}
                meta = json.loads(parts['metadata'].get_payload(decode=True))
                if meta.get('consent') is not True:
                    raise InvalidRequest('consent_required')
                raw = parts['file'].get_payload(decode=True)
                if not raw or len(raw) > 64 * 1024 * 1024:
                    raise InvalidRequest('file_size', 413)
                job_id = start_audio(raw, clean_metadata(meta), int(meta.get('seconds', 60)), int(meta.get('speakers', 0)))
                return self.respond({'id': job_id}, 202)
            data = json.loads(self.body())
            if not isinstance(data, dict):
                raise InvalidRequest()
            if path == '/api/transcript':
                metadata = clean_metadata(data)
                rows = clean_rows(parse_text(str(data.get('text', ''))))
                return self.respond({'rows': rows, 'metadata': metadata, 'partial': False,
                                     'source': 'Текст, введенный пользователем'})
            if path == '/api/jobs/report':
                return self.respond({'id': start_report(data)}, 202)
            match = re.fullmatch(r'/api/protocols/([a-f0-9]{32})', path)
            if match:
                return self.respond(update_protocol(match.group(1), data))
            raise InvalidRequest('not_found', 404)
        except Exception as exc:
            self.handle_error(exc)


def make_server(port=8501):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    actual = server.server_address[1]
    server.allowed_hosts = {'127.0.0.1:%d' % actual, 'localhost:%d' % actual}
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    server = make_server(args.port)
    print('Khattama: http://127.0.0.1:%d' % args.port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
