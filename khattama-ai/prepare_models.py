"""Explicit download phase. This script never opens meeting files."""
import argparse
import hashlib
import json
import os
import shutil
import ssl
import tarfile
import urllib.request
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ASR_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
LLM_REVISION = "7dabda4d13d513e3e842b20f0d435c732f172cbe"
LLM_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"


def download(url, destination):
    import certifi
    if destination.is_file() and destination.stat().st_size > 1024:
        print("Уже загружено:", destination.name, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": "KhattamaAI-model-setup/0.1"})
    print("Загрузка:", destination.name, flush=True)
    with urllib.request.urlopen(request, timeout=120, context=context) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "asr", "diarization", "llm"], default="all")
    args = parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    from huggingface_hub import hf_hub_download, snapshot_download
    if args.only in ("all", "asr"):
        print("1. Whisper small (многоязычная модель)", flush=True)
        snapshot_download("Systran/faster-whisper-small", revision=ASR_REVISION,
                          local_dir=str(MODELS / "whisper-small"),
                          allow_patterns=["*.json", "model.bin", "vocabulary.*"], max_workers=2)
    if args.only in ("all", "diarization"):
        print("2. Модели разделения голосов", flush=True)
        archive = MODELS / "segmentation.tar.bz2"
        if not (MODELS / "segmentation" / "model.onnx").is_file():
            download(RELEASE + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", archive)
            destination = MODELS / "segmentation"
            destination.mkdir(exist_ok=True)
            # Copy selected regular files only; never extract paths or links from an archive.
            with tarfile.open(archive, "r:bz2") as tar:
                for member in tar.getmembers():
                    name = Path(member.name).name
                    if member.isfile() and name in {"model.onnx", "LICENSE", "README.md"}:
                        with tar.extractfile(member) as source, (destination / name).open("wb") as target:
                            shutil.copyfileobj(source, target)
            archive.unlink()
        download(RELEASE + "speaker-recongition-models/nemo_en_titanet_small.onnx",
                 MODELS / "nemo_en_titanet_small.onnx")
    if args.only in ("all", "llm"):
        print("3. Qwen2.5 3B Q4_K_M (языковая модель)", flush=True)
        needed = ["LICENSE"]
        if not (MODELS / LLM_FILE).is_file():
            needed.append(LLM_FILE)
        for name in needed:
            hf_hub_download("Qwen/Qwen2.5-3B-Instruct-GGUF", filename=name,
                            revision=LLM_REVISION, local_dir=str(MODELS / "qwen-download"))
        if not (MODELS / LLM_FILE).is_file():
            (MODELS / "qwen-download" / LLM_FILE).replace(MODELS / LLM_FILE)
        shutil.copy2(MODELS / "qwen-download" / "LICENSE", MODELS / "QWEN_LICENSE")
    print("Формирование контрольных сумм…", flush=True)
    manifest = {"asr_revision": ASR_REVISION, "llm_revision": LLM_REVISION, "files": {}}
    for path in MODELS.rglob("*"):
        if not path.is_file() or ".cache" in path.parts or path.name == "manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["files"][str(path.relative_to(MODELS))] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    (MODELS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Готово. Во время обработки записей интернет не требуется.", flush=True)


if __name__ == "__main__":
    main()
