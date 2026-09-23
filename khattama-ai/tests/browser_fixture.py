"""Isolated browser fixture: synthetic data and stubbed model output only."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server, protocol_store
from core import parse_text
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
work = tempfile.TemporaryDirectory(prefix='khattama-ui-test-')
protocol_store.DEFAULT_PATH = Path(work.name) / 'protocols.sqlite3'
server.model_status = lambda: {'asr': True, 'seg': True, 'voices': True, 'llm': True}
server.decode_audio = lambda raw: np.zeros(16000 * 3, dtype=np.float32)
server.transcribe = lambda *args, **kwargs: ([{'start':0.0,'end':1.0,'text':' Есепті жіберіңіз.'}], ['kk'])
server.diarize = lambda *args, **kwargs: [{'start':0.0,'end':3.0,'speaker_id':'SPEAKER_00'}]
def extraction(rows, anchor=None, roster='', output_language='ru'):
    return {'summary': 'Браузерді тексеруге арналған синтетикалық нәтиже.', 'actions': [
        {'id':'A001','action':'Тексеру есебін жіберу','owner':'Айдана','deadline_raw':'','due_date':'',
         'evidence_ids':[rows[0]['id']], 'note':'','verified':False}]}
server.extract_actions = extraction
today = datetime.now(ZoneInfo('Asia/Almaty')).date()
for i, title in enumerate(['Апталық жоспарлау', 'Өнімді дамыту', 'Жоба қорытындысы']):
    rows = parse_text('Айдана: Тексеру есебін жіберіңіз.')
    action = extraction(rows)['actions'][0]
    action.update(due_date=(today+timedelta(days=i-1)).isoformat(), status='pending')
    bundle = {'title':title+' · сынақ', 'meeting_date':today.isoformat(), 'source':'Текст, введенный пользователем',
              'partial_audio':False,'summary':'Тек интерфейсті тексеруге арналған синтетикалық үлгі.','transcript':rows,
              'actions':[action], 'review_status':'reviewed' if i==1 else 'draft','document_language':'kk'}
    protocol_store.save_protocol(bundle,server.document(bundle))
http = server.make_server(0)
print('http://127.0.0.1:%d' % http.server_address[1],flush=True)
try: http.serve_forever()
finally: http.server_close(); work.cleanup()
