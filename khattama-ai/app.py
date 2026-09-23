import json
from datetime import date
import logging
import sqlite3
from i18n import translate
from protocol_store import save_protocol, list_protocols, load_protocol, display_time
import pandas as pd
import streamlit as st
from core import (ROOT, align_words, decode_audio, diarize, export_docx, extract_actions,
                  fingerprint, parse_text, timestamp, transcribe, wav_bytes)

st.set_page_config(page_title="Хаттама AI", page_icon="📝", layout="wide")
st.markdown("<style>[data-testid='stAppDeployButton'], .stDeployButton, #MainMenu {display:none!important}</style>", unsafe_allow_html=True)
ui_language = st.sidebar.selectbox("Тіл / Язык", ["kk", "ru"], format_func=lambda code: {"kk": "Қазақша", "ru": "Русский"}[code], key="ui_language")
def t(text):
    return translate(text, ui_language)

st.title("Хаттама AI")
st.caption(t("Запись встречи → текст → проверяемые поручения → протокол"))

if "generation" not in st.session_state:
    st.session_state.generation = 0


page = st.sidebar.radio(t("Источник"), ["new", "saved"],
                        format_func=lambda value: t("Новый протокол") if value == "new" else t("Сохранённые протоколы"), key="page")
if page == "saved":
    st.subheader(t("Сохранённые протоколы"))
    st.caption(t("Время Алматы (UTC+5)"))
    try:
        saved = list_protocols()
        if not saved:
            st.info(t("Пока нет сохранённых протоколов. Создайте первый — он сохранится автоматически."))
        else:
            choices = {item["id"]: item for item in saved}
            selected = st.selectbox(t("Выберите протокол"), list(choices),
                format_func=lambda key: display_time(choices[key]["created_at"]) + " · " + choices[key]["title"], key="saved_protocol")
            record = load_protocol(selected)
            bundle = record["bundle"]
            st.subheader(bundle["title"])
            st.caption(t("Сохранено") + ": " + display_time(record["created_at"]) + " · " + t("Обновлено") + ": " + display_time(record["updated_at"]))
            st.write(t("Дата совещания") + ": " + (bundle["meeting_date"] or t("Не указана")))
            st.write(bundle["summary"])
            for action in bundle["actions"]:
                st.write("• " + action["action"] + " — " + (action.get("owner") or t("Не указан")) + " · " + (action.get("due_date") or action.get("deadline_raw") or t("Не указан")))
            st.download_button(t("Скачать протокол DOCX"), record["document"],
                "protocol_" + selected[:8] + ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
            with st.expander(t("Скачать JSON")):
                export = {**bundle, "saved_at": record["created_at"], "updated_at": record["updated_at"]}
                st.download_button(t("Скачать JSON"), json.dumps(export, ensure_ascii=False, indent=2), "protocol_" + selected[:8] + ".json", "application/json")
    except (OSError, ValueError, KeyError, sqlite3.Error):
        logging.exception("Protocol archive could not be loaded")
        st.error(t("Не удалось открыть сохранённые протоколы. Повторите попытку."))
    st.stop()

with st.sidebar:
    st.subheader(t("Совещание"))
    title = st.text_input(t("Название"), st.session_state.get("draft_title", t("Рабочее совещание")))
    st.session_state.draft_title = title
    known_date = st.checkbox(t("Дата совещания известна"), value=st.session_state.get("draft_known_date", False))
    st.session_state.draft_known_date = known_date
    anchor = st.date_input(t("Дата"), value=st.session_state.get("draft_date", date.today()), disabled=not known_date).isoformat() if known_date else None
    if anchor:
        st.session_state.draft_date = date.fromisoformat(anchor)
    roster = st.text_area(t("Участники и упомянутые сотрудники"), value=st.session_state.get("draft_roster", ""), placeholder=t("Имена, должности; можно добавить отсутствующего исполнителя"))
    st.session_state.draft_roster = roster
    st.caption(t("Дату загрузки не используем как дату совещания. Сроки требуют проверки."))
    st.caption(t("Протоколы сохраняются на этом компьютере."))
    if st.button(t("Очистить текущую сессию")):
        version = st.session_state.generation + 1
        for key in list(st.session_state):
            if key not in {"ui_language", "page"}:
                del st.session_state[key]
        st.session_state.generation = version
        st.rerun()

mode = st.radio(t("Источник"), [t("Аудио / видео"), t("Текст для проверки")], horizontal=True)
if mode == t("Аудио / видео"):
    upload = st.file_uploader(t("Загрузите запись до 10 минут"), type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
                              key="upload_%s" % st.session_state.generation)
    c1, c2, c3 = st.columns(3)
    language_label = c1.selectbox(t("Язык"), [t("Русский"), t("Қазақша"), t("Смешанный RU/KZ")])
    language = {t("Русский"): "ru", t("Қазақша"): "kk", t("Смешанный RU/KZ"): "mixed"}[language_label]
    count = c2.number_input(t("Число голосов (0 — определить)"), 0, 12, 0)
    fragment = c3.selectbox(t("Объём обработки"), [t("Первые 60 секунд"), t("Вся запись")])
    notified = st.checkbox(t("Это учебная запись либо участники уведомлены о записи и обработке ИИ"))
    st.caption(t("Сначала проверьте 60 секунд. Если запись озвучена одним человеком, несколько ролей в тексте не означают несколько голосов."))
    if st.button(t("Распознать и разделить голоса"), type="primary", disabled=not upload or not notified):
        try:
            with st.status(t("Обрабатываем запись…"), expanded=True) as status:
                st.write(t("Чтение аудиодорожки"))
                samples = decode_audio(upload.getvalue())
                full_duration = len(samples) / 16000
                partial = fragment == t("Первые 60 секунд") and full_duration > 60
                if partial:
                    samples = samples[:60 * 16000]
                st.write(t("Распознавание речи"))
                bar = st.progress(0.0)
                words, detected = transcribe(samples, language=language, progress=bar.progress)
                st.write(t("Определение говорящих"))
                turns = diarize(samples, int(count))
                rows = align_words(words, turns)
                version = st.session_state.generation + 1
                st.session_state.generation = version
                st.session_state.meeting = {"rows": rows, "samples": samples, "turns": turns,
                    "mode": "Локальное распознавание аудио", "partial": partial,
                    "full_duration": full_duration, "processed_duration": len(samples) / 16000,
                    "detected_languages": detected, "version": version}
                st.session_state.pop("report", None)
                status.update(label=t("Распознавание завершено. Проверьте текст и имена."), state="complete")
        except Exception as exc:
            logging.exception("Meeting processing failed")
            st.error(t("Не удалось обработать данные. Проверьте файл и готовность приложения."))
else:
    st.info(t("Вставьте текст встречи, чтобы подготовить поручения и протокол."))
    example = (ROOT / "examples" / "mixed_meeting.txt").read_text(encoding="utf-8")
    raw_text = st.text_area(t("Каждая реплика: Имя: текст"), value=example, height=210,
                            key="text_%s" % st.session_state.generation)
    if st.button(t("Использовать этот текст")):
        rows = parse_text(raw_text)
        if rows:
            version = st.session_state.generation + 1
            st.session_state.generation = version
            st.session_state.meeting = {"rows": rows, "mode": "Текст, введенный пользователем",
                                        "partial": False, "version": version}
            st.session_state.pop("report", None)
        else:
            st.warning(t("Введите хотя бы одну реплику."))

meeting = st.session_state.get("meeting")
if not meeting:
    st.markdown(t("**Начните с короткого фрагмента.** После обработки подтвердите имена говорящих, проверьте текст и сформируйте поручения."))
    st.stop()

st.divider()
st.subheader(t("1. Участники и текст"))
st.caption(t("Текущий результат: ") + t(meeting["mode"]))
if meeting["partial"]:
    st.warning(t("Обработаны только первые 60 секунд. Экспорт будет помечен как фрагмент."))

rows = [dict(r) for r in meeting["rows"]]
if "samples" in meeting:
    st.audio(wav_bytes(meeting["samples"]), format="audio/wav")
    speaker_ids = sorted({r["speaker_id"] for r in rows})
    with st.expander(t("Подтвердить имена голосов"), expanded=True):
        mapping = {}
        for sid in speaker_ids:
            left, right = st.columns([1, 2])
            mapping[sid] = left.text_input(sid, value=next((r.get("speaker", "") for r in rows if r["speaker_id"] == sid), ""), placeholder=t("Имя после прослушивания"),
                                           key="speaker_%s_%s" % (meeting["version"], sid))
            candidates = [r for r in rows if r["speaker_id"] == sid]
            clip = max(candidates, key=lambda r: r["end"] - r["start"])
            start, end = int(clip["start"] * 16000), int(min(clip["end"], clip["start"] + 12) * 16000)
            right.audio(wav_bytes(meeting["samples"][start:end]), format="audio/wav")
        for r in rows:
            r["speaker"] = mapping[r["speaker_id"]].strip() or r["speaker_id"]

if any(r.get("speaker_review") for r in rows):
    st.warning(t("Есть неуверенная привязка слов к голосам или наложение речи. Проверьте отмеченные реплики."))

frame = pd.DataFrame([{"id": r["id"], "speaker": r.get("speaker", r["speaker_id"]),
                       "text": r["text"], "time": timestamp(r["start"]),
                       "review": r.get("speaker_review", False)} for r in rows])
editor_key = "transcript_%s_%s" % (meeting["version"], fingerprint(frame.to_dict("records"))[:12])
edited = st.data_editor(frame, hide_index=True, use_container_width=True, disabled=["id", "time", "review"],
    column_config={"id": "ID", "speaker": t("Кто говорит"), "text": st.column_config.TextColumn(t("Реплика"), width="large"),
                   "time": t("Начало"), "review": t("Проверить голос")}, key=editor_key)
for row, edit in zip(rows, edited.to_dict("records")):
    row["speaker"] = str(edit["speaker"] or row["speaker_id"])
    row["text"] = str(edit["text"] or "")
meeting["rows"] = rows

current_fp = fingerprint({"rows": rows, "anchor": anchor, "roster": roster})
if st.button(t("2. Выделить поручения и подготовить саммари"), type="primary"):
    try:
        with st.spinner(t("Готовим протокол. Это может занять несколько минут…")):
            result = extract_actions(rows, anchor, roster)
            st.session_state.report = {"result": result, "fingerprint": current_fp}
    except Exception as exc:
        logging.exception("Protocol generation failed")
        st.error(t("Не удалось обработать данные. Проверьте файл и готовность приложения."))

report = st.session_state.get("report")
if not report:
    transcript_text = "\n".join("[%s %s] %s: %s" % (r["id"], timestamp(r["start"]), r["speaker"], r["text"]) for r in rows)
    st.download_button(t("Скачать транскрипт TXT"), transcript_text, "transcript.txt", "text/plain")
    st.stop()
if report["fingerprint"] != current_fp:
    st.warning(t("Текст, имена или дата изменились. Сформируйте поручения заново, чтобы экспорт соответствовал источнику."))
    st.stop()

st.subheader(t("3. Проверить протокол"))
st.caption(t("Результат модели — черновик. Сверьте каждое поручение с репликами-основаниями."))
summary = st.text_area(t("Саммари"), value=report["result"]["summary"], height=130, key="summary_" + current_fp)
records = [{**a, "evidence_ids": ", ".join(a["evidence_ids"])} for a in report["result"]["actions"]]
columns = ["id", "action", "owner", "deadline_raw", "due_date", "evidence_ids", "note", "verified"]
action_frame = pd.DataFrame(records, columns=columns)
action_frame["verified"] = action_frame["verified"].astype(bool)
updated = st.data_editor(action_frame, num_rows="dynamic", use_container_width=True, hide_index=True,
    disabled=["id"], key="actions_" + current_fp + fingerprint(records)[:12],
    column_config={"id": "ID", "action": st.column_config.TextColumn(t("Что сделать"), width="large"),
        "owner": t("Ответственный"), "deadline_raw": t("Срок в речи"), "due_date": t("Дата YYYY-MM-DD"),
        "evidence_ids": t("ID реплик через запятую"), "note": t("Уточнения"), "verified": st.column_config.CheckboxColumn(t("Проверено"))})

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
    st.error(t("Исправьте даты на YYYY-MM-DD или очистите поле: ") + ", ".join(invalid_dates))
with st.expander(t("Показать основание поручения"), expanded=True):
    if actions:
        selected = st.selectbox(t("Поручение"), range(len(actions)), format_func=lambda i: actions[i]["id"] + " — " + actions[i]["action"])
        evidence = actions[selected]["evidence_ids"]
        if not evidence:
            st.warning(t("Нет подтвержденной ссылки на реплику. Проверьте поручение вручную."))
        for row in rows:
            if row["id"] in evidence:
                st.write("[%s · %s] %s: %s" % (row["id"], timestamp(row["start"]), row["speaker"], row["text"]))
                if "samples" in meeting:
                    start, end = max(0, int((row["start"] - 0.3) * 16000)), int((row["end"] + 0.3) * 16000)
                    st.audio(wav_bytes(meeting["samples"][start:end]), format="audio/wav")

report["result"]["summary"] = summary
report["result"]["actions"] = actions
bundle = {"title": title, "meeting_date": anchor, "source": meeting["mode"], "partial_audio": meeting["partial"],
          "summary": summary, "actions": actions, "transcript": rows, "review_status": "draft", "document_language": ui_language}
if not invalid_dates:
    doc = export_docx(title, anchor, summary, actions, rows, t(meeting["mode"]), meeting["partial"], ui_language=ui_language)
    try:
        saved = save_protocol(bundle, doc, record_id=report.get("storage_id"))
        report["storage_id"] = saved["id"]
        st.success(t("Протокол сохранён автоматически") + " · " + display_time(saved["created_at"]))
        st.caption(t("Изменения сохраняются автоматически. Дата сохранения не меняет дату совещания."))
    except (OSError, sqlite3.Error):
        logging.exception("Protocol could not be saved")
        st.error(t("Не удалось сохранить протокол. Проверьте свободное место и повторите попытку."))
        st.button(t("Повторить сохранение"))
    st.download_button(t("Скачать протокол DOCX"), doc, "meeting_protocol.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
    with st.expander(t("Скачать JSON")):
        st.download_button(t("Скачать JSON"), json.dumps(bundle, ensure_ascii=False, indent=2), "meeting.json", "application/json")
