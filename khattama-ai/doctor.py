import importlib
import platform
from core import model_status

print('Python:', platform.python_version(), '|', platform.system(), platform.machine())
failed = []
for name in ['streamlit', 'numpy', 'av', 'faster_whisper', 'sherpa_onnx', 'llama_cpp', 'docx']:
    try:
        module = importlib.import_module(name)
        print('OK', name, getattr(module, '__version__', ''))
    except Exception as exc:
        print('ERROR', name, str(exc))
        failed.append(name)
for name, ready in model_status().items():
    print('MODEL', name, 'OK' if ready else 'не скачана')
if failed:
    raise SystemExit('Ошибка загрузки библиотек: ' + ', '.join(failed))
