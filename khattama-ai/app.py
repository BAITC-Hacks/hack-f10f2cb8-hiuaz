import json
from datetime import date
import pandas as pd
import streamlit as st
from core import (ROOT, align_words, decode_audio, diarize, export_docx, extract_actions,
                  fingerprint, model_status, parse_text, timestamp, transcribe, wav_bytes)

st.set_page_config(page_title="Хаттама AI", page_icon="📝", layout="wide")
st.title("Хаттама AI")
st.caption("Запись встречи → текст → проверяемые поручения → протокол")

if "generation" not in st.session_state:
    st.session_state.generation = 0

with st.sidebar:
    st.subheader("Совещание")
    title = st.text_input("Название", "Рабочее совещание")
    known_date = st.checkbox("Дата совещания известна", value=False)
    anchor = st.date_input("Дата", value=date.today(), disabled=not known_date).isoformat() if known_date else None
    roster = st.text_area("Участники и упомянутые сотрудники", placeholder="Имена, должности; можно добавить отсутствующего исполнителя")
    st.caption("Дату загрузки не используем как дату совещания. Сроки требуют проверки.")
    with st.expander("Готовность моделей"):
        for label, ready in model_status().items():
            st.write(("✓ " if ready else "○ ") + label)
        st.code(".venv/bin/python prepare_models.py", language="bash")
    st.caption("Обработка на этом компьютере. Первый запуск требует загрузки весов моделей.")
    if st.button("Очистить текущую сессию"):
        version = st.session_state.generation + 1
        st.session_state.clear()
        st.session_state.generation = version
        st.rerun()

mode = st.radio("Источник", ["Аудио / видео", "Текст для проверки"], horizontal=True)
if mode == "Аудио / видео":
    upload = st.file_uploader("Загрузите запись до 10 минут", type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
                              key="upload_%s" % st.session_state.generation)
    c1, c2, c3 = st.columns(3)
    language_label = c1.selectbox("Язык", ["Русский", "Қазақша", "Смешанный RU/KZ"])
    language = {"Русский": "ru", "Қазақша": "kk", "Смешанный RU/KZ": "mixed"}[language_label]
    count = c2.number_input("Число голосов (0 — определить)", 0, 12, 0)
    fragment = c3.selectbox("Объём обработки", ["Первые 60 секунд", "Вся запись"])
    notified = st.checkbox("Это учебная запись либо участники уведомлены о записи и обработке ИИ")
    st.caption("Сначала проверьте 60 секунд. Если запись озвучена одним человеком, несколько ролей в тексте не означают несколько голосов.")
    if st.button("Распознать и разделить голоса", type="primary", disabled=not upload or not notified):
        try:
            with st.status("Обработка на CPU…", expanded=True) as status:
                st.write("Чтение аудиодорожки")
                samples = decode_audio(upload.getvalue())
                full_duration = len(samples) / 16000
                partial = fragment == "Первые 60 секунд" and full_duration > 60
                if partial:
                    samples = samples[:60 * 16000]
                st.write("Распознавание речи")
                bar = st.progress(0.0)
                words, detected = transcribe(samples, language=language, progress=bar.progress)
                st.write("Определение говорящих")
                turns = diarize(samples, int(count))
                rows = align_words(words, turns)
                version = st.session_state.generation + 1
                st.session_state.generation = version
                st.session_state.meeting = {"rows": rows, "samples": samples, "turns": turns,
                    "mode": "Локальное распознавание аудио", "partial": partial,
                    "full_duration": full_duration, "processed_duration": len(samples) / 16000,
                    "detected_languages": detected, "version": version}
                st.session_state.pop("report", None)
                status.update(label="Распознавание завершено. Проверьте текст и имена.", state="complete")
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("Этот режим проверяет извлечение поручений из текста. Он не выполняет распознавание и диаризацию.")
    example = (ROOT / "examples" / "mixed_meeting.txt").read_text(encoding="utf-8")
    raw_text = st.text_area("Каждая реплика: Имя: текст", value=example, height=210,
                            key="text_%s" % st.session_state.generation)
    if st.button("Использовать этот текст"):
        rows = parse_text(raw_text)
        if rows:
            version = st.session_state.generation + 1
            st.session_state.generation = version
            st.session_state.meeting = {"rows": rows, "mode": "Текст, введенный пользователем",
                                        "partial": False, "version": version}
            st.session_state.pop("report", None)
        else:
            st.warning("Введите хотя бы одну реплику.")

meeting = st.session_state.get("meeting")
if not meeting:
    st.markdown("**Начните с короткого фрагмента.** После обработки подтвердите имена говорящих, проверьте текст и сформируйте поручения.")
    st.stop()

st.divider()
st.subheader("1. Участники и текст")
st.caption("Текущий результат: " + meeting["mode"])
if meeting["partial"]:
    st.warning("Обработаны только первые 60 секунд. Экспорт будет помечен как фрагмент.")

rows = [dict(r) for r in meeting["rows"]]
if "samples" in meeting:
    st.audio(wav_bytes(meeting["samples"]), format="audio/wav")
    speaker_ids = sorted({r["speaker_id"] for r in rows})
    with st.expander("Подтвердить имена голосов", expanded=True):
        mapping = {}
        for sid in speaker_ids:
            left, right = st.columns([1, 2])
            mapping[sid] = left.text_input(sid, value="", placeholder="Имя после прослушивания",
                                           key="speaker_%s_%s" % (meeting["version"], sid))
            candidates = [r for r in rows if r["speaker_id"] == sid]
            clip = max(candidates, key=lambda r: r["end"] - r["start"])
            start, end = int(clip["start"] * 16000), int(min(clip["end"], clip["start"] + 12) * 16000)
            right.audio(wav_bytes(meeting["samples"][start:end]), format="audio/wav")
        for r in rows:
            r["speaker"] = mapping[r["speaker_id"]].strip() or r["speaker_id"]

if any(r.get("speaker_review") for r in rows):
    st.warning("Есть неуверенная привязка слов к голосам или наложение речи. Проверьте отмеченные реплики.")

frame = pd.DataFrame([{"id": r["id"], "speaker": r.get("speaker", r["speaker_id"]),
                       "text": r["text"], "time": timestamp(r["start"]),
                       "review": r.get("speaker_review", False)} for r in rows])
editor_key = "transcript_%s_%s" % (meeting["version"], fingerprint(frame.to_dict("records"))[:12])
edited = st.data_editor(frame, hide_index=True, use_container_width=True, disabled=["id", "time", "review"],
    column_config={"id": "ID", "speaker": "Кто говорит", "text": st.column_config.TextColumn("Реплика", width="large"),
                   "time": "Начало", "review": "Проверить голос"}, key=editor_key)
for row, edit in zip(rows, edited.to_dict("records")):
    row["speaker"] = str(edit["speaker"] or row["speaker_id"])
    row["text"] = str(edit["text"] or "")

current_fp = fingerprint({"rows": rows, "anchor": anchor, "roster": roster})
if st.button("2. Выделить поручения и подготовить саммари", type="primary"):
    try:
        with st.spinner("Локальная модель анализирует реплики. На CPU это может занять несколько минут…"):
            result = extract_actions(rows, anchor, roster)
            st.session_state.report = {"result": result, "fingerprint": current_fp}
    except Exception as exc:
        st.error(str(exc))

report = st.session_state.get("report")
if not report:
    transcript_text = "\n".join("[%s %s] %s: %s" % (r["id"], timestamp(r["start"]), r["speaker"], r["text"]) for r in rows)
    st.download_button("Скачать транскрипт TXT", transcript_text, "transcript.txt", "text/plain")
    st.stop()
if report["fingerprint"] != current_fp:
    st.warning("Текст, имена или дата изменились. Сформируйте поручения заново, чтобы экспорт соответствовал источнику.")
    st.stop()

st.subheader("3. Проверить протокол")
st.caption("Результат модели — черновик. Сверьте каждое поручение с репликами-основаниями.")
summary = st.text_area("Саммари", value=report["result"]["summary"], height=130, key="summary_" + current_fp)
records = [{**a, "evidence_ids": ", ".join(a["evidence_ids"])} for a in report["result"]["actions"]]
columns = ["id", "action", "owner", "deadline_raw", "due_date", "evidence_ids", "note", "verified"]
action_frame = pd.DataFrame(records, columns=columns)
action_frame["verified"] = action_frame["verified"].astype(bool)
updated = st.data_editor(action_frame, num_rows="dynamic", use_container_width=True, hide_index=True,
    disabled=["id"], key="actions_" + current_fp,
    column_config={"id": "ID", "action": st.column_config.TextColumn("Что сделать", width="large"),
        "owner": "Ответственный", "deadline_raw": "Срок в речи", "due_date": "Дата YYYY-MM-DD",
        "evidence_ids": "ID реплик через запятую", "note": "Уточнения", "verified": st.column_config.CheckboxColumn("Проверено")})

valid_ids = {r["id"] for r in rows}
actions, invalid_dates = [], []
for i, item in enumerate(updated.to_dict("records")):
    a = {}
    for key in columns:
        value = item.get(key)
        if key == "verified":
            a[key] = False if value is None or pd.isna(value) else bool(value)
        else:
            a[key] = "" if value is None or pd.isna(value) else str(value).strip()
    if not a["action"]:
        continue
    a["id"] = a["id"] or "USER_%03d" % (i + 1)
    a["evidence_ids"] = [x.strip() for x in a["evidence_ids"].split(",") if x.strip() in valid_ids]
    if a["due_date"]:
        try:
            date.fromisoformat(a["due_date"])
        except ValueError:
            invalid_dates.append(a["id"])
    actions.append(a)

if invalid_dates:
    st.error("Исправьте даты на YYYY-MM-DD или очистите поле: " + ", ".join(invalid_dates))
with st.expander("Показать основание поручения", expanded=True):
    if actions:
        selected = st.selectbox("Поручение", range(len(actions)), format_func=lambda i: actions[i]["id"] + " — " + actions[i]["action"])
        evidence = actions[selected]["evidence_ids"]
        if not evidence:
            st.warning("Нет подтвержденной ссылки на реплику. Проверьте поручение вручную.")
        for row in rows:
            if row["id"] in evidence:
                st.write("[%s · %s] %s: %s" % (row["id"], timestamp(row["start"]), row["speaker"], row["text"]))
                if "samples" in meeting:
                    start, end = max(0, int((row["start"] - 0.3) * 16000)), int((row["end"] + 0.3) * 16000)
                    st.audio(wav_bytes(meeting["samples"][start:end]), format="audio/wav")

bundle = {"title": title, "meeting_date": anchor, "source": meeting["mode"], "partial_audio": meeting["partial"],
          "summary": summary, "actions": actions, "transcript": rows, "review_status": "draft"}
left, right = st.columns(2)
left.download_button("Скачать JSON", json.dumps(bundle, ensure_ascii=False, indent=2), "meeting.json", "application/json", disabled=bool(invalid_dates))
if not invalid_dates:
    doc = export_docx(title, anchor, summary, actions, rows, meeting["mode"], meeting["partial"])
    right.download_button("Скачать протокол DOCX", doc, "meeting_protocol.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
