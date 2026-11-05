"""NSW Defect Detective - Streamlit front end.

What changed in this version (search for "BUG FIX" / "NEW" in the code):
  * NEW  Sidebar case lists: "Unsolved cases" (continue one) and "Case reports" (PDF per case).
         The per-message "Export this case" buttons and the bottom "This case report" button
         are gone; the Machine history report stays at the bottom.
  * NEW  "Continue troubleshooting this issue" after a report: the operator reports what they
         tried, the AI gets the earlier causes as ruled-out/known, and pages it already used
         are withheld so it has to work from new evidence.
  * NEW  Message router: off-topic -> organised refusal, machine question -> documentation
         answer, greeting -> guidance, defect -> diagnosis.
  * NEW  Fleet memory ranked by similarity: this machine's closed AND unresolved cases, plus
         confirmed fixes on other machines with the same pump.
  * NEW  "Mark as fixed" asks WHICH cause it was, so fleet memory records cause + fix.
  * BUG FIX dark mode / audience / mode needed two clicks (widget re-created every rerun).
  * BUG FIX chat page showed every case the machine ever had; it now shows the current case.
  * BUG FIX the uploaded log's telemetry was injected into every answer message.
  * BUG FIX DEFAULTS["case"] was shallow-copied, sharing its lists between cases.
"""
import datetime as _dt
import os
import re
import tempfile
import urllib.parse

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from database import (add_machine, get_all_machines, get_case_ids, get_chat_history,
                      get_fleet_resolutions, get_machine, get_resolution_for_case,
                      get_resolutions, init_db, next_case_id, save_chat_message,
                      save_resolution)
from model import (ANSWER_HEADING, DISCOVERY_QUESTIONS, clean_questions, LLM_MODEL, MAX_FORMAT_RETRIES, MAX_INPUT_CHARS,
                   MAX_INQUIRY_ROUNDS, MAX_RAG_CHARS, MAX_CAUSE_ROWS, MIN_CAUSE_ROWS,
                   PLATFORM_ALIASES, approx_tokens, build_continuation_block,
                   build_diagnosis_instruction, build_fleet_memory, build_history_messages,
                   build_inquiry_instruction, build_question_instruction, chunk_key,
                   chunk_text, cited_pages, clip, extract_causes, fix_citations,
                   format_chunks, generate_report, get_hardware_blueprint, greeting_reply,
                   hardware_inventory, hardware_tags, inquiry_reminder, is_non_answer,
                   is_out_of_scope, load_prompt_template, make_llm, normalize_name,
                   off_topic_reply, open_vector_store, prompt_budget, render_template,
                   renumber_questions, repeated_causes, report_sections, retrieval_report,
                   retrieve_relevant, retry_seed, route_message, sanitize_field,
                   source_names, validate_inquiry)
from report_pdf import repair_markdown
from reports import (build_case_report_pdf, build_machine_history_pdf, case_summary,
                     is_diagnosis, split_case_full)

try:
    from log_parser import extract_log_telemetry
except ImportError:
    extract_log_telemetry = None

try:
    from streamlit_mic_recorder import speech_to_text
except ImportError:
    speech_to_text = None

init_db()
st.set_page_config(page_title="NSW Defect Detective", page_icon="🔬", layout="wide")

DARK_CSS = """
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] { background:#11141a; color:#e6e8ee; }
[data-testid="stSidebar"] { border-right:1px solid #232733; }
h1,h2,h3,h4,h5,h6,p,li,label,span,div[data-testid="stMarkdownContainer"] { color:#e6e8ee; }
[data-testid="stChatMessage"] { background:#171b23; border:1px solid #232733; border-radius:10px; }
table { border-color:#2b3140 !important; }
thead tr th { background:#14213D !important; color:#FCA311 !important; }
tbody tr:nth-child(even) td { background:#171b23 !important; }
tbody tr td { border-color:#2b3140 !important; color:#e6e8ee !important; }
textarea, input, .stTextArea textarea { background:#171b23 !important; color:#e6e8ee !important; border-color:#2b3140 !important; }
button[kind], .stButton button, .stDownloadButton button { background:#1b212c !important; color:#e6e8ee !important; border-color:#2b3140 !important; }
[data-testid="stExpander"] { background:#171b23; border-color:#232733; }
code { background:#1b212c !important; color:#FCA311 !important; }
a { color:#7fb0ff !important; }
/* Fix for File Uploader dropzone */
[data-testid="stFileUploaderDropzone"] { background:#171b23 !important; border: 1px dashed #2b3140 !important; }
[data-testid="stFileUploaderDropzone"] * { color:#e6e8ee !important; }
/* Fix for external embedded components like Mic Recorder */
iframe { filter: invert(0.9) hue-rotate(180deg); }
</style>
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = next((p for p in (os.path.join(BASE_DIR, "nsw_logo.png"),
                              os.path.join(BASE_DIR, "logo.png")) if os.path.exists(p)), None)
if LOGO_PATH:
    try:
        st.logo(LOGO_PATH)
    except Exception:
        pass

PHASE_LABEL = {"AWAIT_SYMPTOM": "waiting for a symptom", "AWAIT_ANSWERS": "follow-up questions",
               "LOCKED": "report ready", "AWAIT_UPDATE": "continuing - waiting for your update"}


def new_case() -> dict:
    """BUG FIX: dict(DEFAULTS["case"]) was a shallow copy - the transcript/cached lists were
    shared objects. Every case now gets fresh containers."""
    return {"symptom": "", "raw_symptom": "", "transcript": [], "pending_q": "",
            "cached": [], "telemetry_note": "", "telemetry_digest": "", "rounds": 0,
            "reports": [], "updates": [], "used_keys": [], "memory": None}


DEFAULTS = {
    "current_page": "Landing",
    "active_machine": None,
    "draft_text": "",
    "messages": [],
    "chat_phase": "AWAIT_SYMPTOM",
    "case_id": 0,
    "show_fix_input": False,
    "pdf_bytes": None,
    "pdf_name": "report.pdf",
    "case_pdf": None,
    "last_notes": [],
    "tone": "Maintenance Tech",
    "mode": "Deep",
    "dark": False,
    "is_generating": False,
    "pending_query": "",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("case", new_case())

# BUG FIX (dark mode needed a double click): the widgets used to be created with
# value=st.session_state.X and no key. Changing `value` changes the widget's identity, so
# Streamlit threw the click away and rebuilt the widget, and it took a second click to
# stick. The widgets are now bound by key. The line below stops Streamlit from deleting
# those keys while the Landing page (which has no sidebar) is showing.
for _k in ("dark", "tone", "mode"):
    st.session_state[_k] = st.session_state[_k]


@st.cache_resource(show_spinner="Connecting to the vector store...")
def load_ai_backend():
    store = open_vector_store()
    return store, make_llm("ask"), make_llm("report"), make_llm("route")


@st.cache_data(show_spinner=False)
def load_blueprint(platform: str, pump: str):
    store = load_ai_backend()[0]
    return get_hardware_blueprint(store, platform, pump)


@st.cache_data(show_spinner=False)
def parse_log(raw: bytes, name: str):
    if extract_log_telemetry is None:
        return None, "log_parser.py is not available, so the uploaded log was ignored."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        return extract_log_telemetry(tmp_path), None
    except Exception as exc:
        return None, f"Could not parse that log: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def local_time(ts) -> str:
    """SQLite CURRENT_TIMESTAMP is UTC; show the operator local time."""
    try:
        t = _dt.datetime.fromisoformat(str(ts)).replace(tzinfo=_dt.timezone.utc)
        return t.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts or "")[:16]


def short(text, n=90) -> str:
    text = re.sub(r"\s+", " ", re.sub(r">\s*📊.*", "", text or "", flags=re.S)).strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def telemetry_line(telemetry) -> str:
    """One line, shown in the chat bubble."""
    if not telemetry:
        return ""
    alarms = ", ".join((telemetry.get("recent_alarms") or [])[-3:]) or "none recorded"
    return (f"status {telemetry.get('status', 'UNKNOWN')} | material "
            f"{telemetry.get('material', '?')} | nozzle {telemetry.get('nozzle', '?')} | "
            f"{telemetry.get('drift_summary', 'no drift summary')} | last alarms: {alarms}")


def telemetry_mismatch(telemetry, m) -> str:
    if not telemetry:
        return ""
    issues = []
    lp, lm = str(telemetry.get("pump") or ""), str(telemetry.get("model") or "")
    if lp and lp != "Unknown" and normalize_name(m["pump"]) not in normalize_name(lp) \
            and normalize_name(lp) not in normalize_name(m["pump"]):
        issues.append(f"log pump is '{lp}' but this machine is registered with '{m['pump']}'")
    if lm and lm != "Unknown" and m["platform"] not in hardware_tags(PLATFORM_ALIASES, lm):
        issues.append(f"log model is '{lm}' but this machine is a '{m['platform']}'")
    return "; ".join(issues)


def telemetry_digest(telemetry, m) -> str:
    """Multi-line log summary for the model (the whole log is too big for the context)."""
    if not telemetry:
        return ""
    digest = telemetry.get("digest") or telemetry_line(telemetry)
    mismatch = telemetry_mismatch(telemetry, m)
    if mismatch:
        digest += f"\nWARNING: {mismatch}. The log may belong to another machine - say so."
    return digest


def build_llm_messages(tone, machine, results, case, instruction, blueprint,
                       fleet_memory="", kind="report"):
    lines = ["=== CASE FILE ===", f"Reported symptom: {case.get('symptom', '')}"]
    for i, (q, a) in enumerate(case.get("transcript") or [], 1):
        lines.append(f"\nFollow-up round {i}, questions asked:\n{q}")
        lines.append(f"Operator's answer: {a}")
    if case.get("telemetry_digest"):
        lines.append("\n=== MACHINE LOG (uploaded by the operator) ===\n"
                     f"{case['telemetry_digest']}\n"
                     "Read this like an engineer: correlate the alarms and the drift with the "
                     "symptom, and cite it as 'Machine log' when it supports a cause.")
    elif case.get("telemetry_note"):
        lines.append(f"\nMachine log telemetry: {case['telemetry_note']}")
    if case.get("updates"):
        lines.append("\n=== OPERATOR UPDATES (after the earlier report(s)) ===")
        lines += [f"Update {i}: {u}" for i, u in enumerate(case["updates"], 1)]
    if fleet_memory:
        lines.append(fleet_memory)
    case_text = "\n".join(lines)

    # NEW: fit the prompt into the context window instead of letting Ollama silently drop
    # the start of it (which is the system prompt with all the rules).
    budget = prompt_budget("ask" if kind == "ask" else "report")
    template = load_prompt_template()
    system_text = ""
    for rag_max, bp_max in ((MAX_RAG_CHARS, None), (3800, 2600), (2800, 2000), (2000, 1400)):
        bp = blueprint if bp_max is None else clip(blueprint, bp_max)
        system_text = render_template(template, {
            "tone": sanitize_field(tone, "Maintenance Tech"),
            "machine_platform": sanitize_field(machine["platform"], "unknown platform"),
            "pump_type": sanitize_field(machine["pump"], "unknown pump"),
            "hardware_blueprint": bp,
            "rag_context": format_chunks(results, rag_max),
        })
        if approx_tokens(system_text + case_text + instruction) <= budget:
            break
    return [SystemMessage(content=system_text),
            HumanMessage(content=f"{case_text}\n\n{instruction}")]


def make_stream_fn(llm, placeholder):
    def stream_fn(messages):
        out = ""
        placeholder.markdown("🤔 _Thinking…_")
        for chunk in llm.stream(messages):
            out += chunk_text(chunk.content)
            placeholder.markdown(repair_markdown(out) + " ▌")
        out = out.strip()
        placeholder.markdown(repair_markdown(out))
        return out
    return stream_fn


def stream_inquiry(llm, messages, placeholder, round_no: int):
    text = make_stream_fn(llm, placeholder)(messages)
    problems = validate_inquiry(text)
    attempt = 0
    while problems and attempt < MAX_FORMAT_RETRIES:
        attempt += 1
        st.warning(f"Formatting anomaly ({'; '.join(problems)}). Retrying "
                   f"{attempt}/{MAX_FORMAT_RETRIES}...")
        retry = list(messages[:-1]) + [HumanMessage(
            content=f"{messages[-1].content}\n\n{inquiry_reminder(round_no)}")]
        text = make_stream_fn(make_llm("ask", retry_seed()), placeholder)(retry)
        problems = validate_inquiry(text)
    # BUG FIX: keep only the actual questions (numbered from 1) - drops any prompt text the
    # model recited after them. Falls back to the plain renumbered text if nothing parses.
    max_q = DISCOVERY_QUESTIONS if round_no <= 1 else 3
    text = clean_questions(text, max_q) or renumber_questions(text)
    text = repair_markdown(text)
    placeholder.markdown(text)
    return text


def record(serial, role, content):
    row_id = save_chat_message(serial, role, content, st.session_state.case_id)
    st.session_state.messages.append({
        "id": row_id if row_id is not None else -len(st.session_state.messages) - 1,
        "role": role, "content": content, "case_id": st.session_state.case_id})
    st.session_state.pdf_bytes = None


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def collect_cases(serial, include_reports=False):
    out = []
    for meta in get_case_ids(serial):
        msgs = get_chat_history(serial, case_id=meta["case_id"])
        symptom, causes = case_summary(msgs)
        if not symptom and not causes:
            continue
        res = get_resolution_for_case(serial, meta["case_id"])
        entry = {"case_id": meta["case_id"], "started": meta["started"],
                 "symptom": symptom, "causes": causes,
                 "fix": (res or {}).get("fix"), "fixed_at": (res or {}).get("timestamp")}
        if include_reports:
            entry["diagnosis"] = next((x["content"] for x in reversed(msgs)
                                       if is_diagnosis(x["content"])), "")
        out.append(entry)
    return out


def rebuild_case(msgs):
    """Turn a stored case back into live session state so it can be continued."""
    case = new_case()
    full = split_case_full(msgs)
    if not full["inputs"]:
        return case, "AWAIT_SYMPTOM"
    first = full["inputs"][0]
    tele = re.search(r">\s*📊\s*\*\*Telemetry Injected:\*\*\s*(.+)", first, re.S)
    raw = re.sub(r"\n*>\s*📊.*", "", first, flags=re.S).strip()
    note = tele.group(1).strip() if tele else ""
    case.update(raw_symptom=raw, telemetry_note=note,
                symptom=raw + (f"\nTelemetry: {note}" if note else ""),
                transcript=list(full["rounds"]),
                rounds=len(full["rounds"]) + (1 if full["pending_q"] else 0),
                reports=list(full["reports"]),
                updates=[u for _, u in full["updates"]])
    for rep in full["reports"]:
        case["used_keys"] += sorted(cited_pages(rep))
    if full["reports"]:
        return case, "LOCKED"
    if full["pending_q"]:
        case["pending_q"] = full["pending_q"]
        return case, "AWAIT_ANSWERS"
    return case, "AWAIT_SYMPTOM"


def go_to_chat(serial, platform, pump):
    st.session_state.active_machine = {"serial": serial, "platform": platform, "pump": pump}
    st.session_state.case_id = next_case_id(serial)
    # BUG FIX: used to load the machine's ENTIRE history into the chat. Past cases now live
    # in the sidebar; the chat shows the case being worked on.
    st.session_state.messages = []
    st.session_state.chat_phase = "AWAIT_SYMPTOM"
    st.session_state.case = new_case()
    st.session_state.pdf_bytes = None
    st.session_state.case_pdf = None
    st.session_state.last_notes = []
    st.session_state.show_fix_input = False
    st.session_state.current_page = "Chat"
    st.rerun()


def resume_case(m, case_id):
    msgs = get_chat_history(m["serial"], case_id=case_id)
    case, phase = rebuild_case(msgs)
    st.session_state.case_id = case_id
    st.session_state.messages = msgs
    st.session_state.case = case
    st.session_state.chat_phase = phase
    st.session_state.last_notes = []
    st.session_state.show_fix_input = False
    st.rerun()


def start_new_case(m, note="[SYSTEM] Case set aside. Starting a new case on this machine."):
    if st.session_state.messages:
        record(m["serial"], "assistant", note)
    st.session_state.case_id = next_case_id(m["serial"])
    st.session_state.chat_phase = "AWAIT_SYMPTOM"
    st.session_state.case = new_case()
    st.session_state.messages = []
    st.session_state.last_notes = []
    st.session_state.show_fix_input = False
    st.rerun()


def export_case(m, case_id):
    msgs = get_chat_history(m["serial"], case_id=case_id)
    res = get_resolution_for_case(m["serial"], case_id)
    pdf = build_case_report_pdf(
        m["serial"], msgs, platform=m["platform"], pump=m["pump"],
        tone=st.session_state.tone, mode=st.session_state.mode, logo_path=LOGO_PATH,
        case_id=case_id, resolution=res, telemetry_note="")
    st.session_state.case_pdf = {"case_id": case_id, "bytes": pdf,
                                 "name": f"Case_{m['serial']}_{case_id}.pdf"}


def case_memory(m, store, case) -> str:
    """Similarity-ranked history of this machine + same-pump fixes across the fleet."""
    if case.get("memory") is None:
        own = [c for c in collect_cases(m["serial"]) if c["case_id"] != st.session_state.case_id]
        fleet = get_fleet_resolutions(m["pump"], exclude_serial=m["serial"])
        emb = getattr(store, "embeddings", None)
        embed_fn = getattr(emb, "embed_documents", None)
        case["memory"] = build_fleet_memory(case.get("raw_symptom") or case.get("symptom", ""),
                                            own, fleet, embed_fn, m["pump"])
    return case["memory"]


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_landing_page():
    st.title("🏭 NSW Fleet Diagnostic Portal")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Register New Machine")
        with st.form("new_machine_form"):
            new_serial = st.text_input("Serial Number", placeholder="e.g., ABC-123-XYZ")
            new_platform = st.selectbox("Machine Platform", list(PLATFORM_ALIASES))
            new_pump = st.selectbox("Fluid Delivery System",
                                    ["SynchroPULSE-MICRO", "PistonPRO 4+",
                                     "HeliMASTER-3", "Time Pressure System"])
            if st.form_submit_button("Register & Troubleshoot"):
                serial = new_serial.strip().upper()
                if not re.match(r"^[A-Z0-9]{3}-[A-Z0-9]{3}-[A-Z0-9]{3}$", serial):
                    st.error("Invalid format. Use 9 characters separated by dashes "
                             "(e.g., ABC-123-XYZ).")
                elif add_machine(serial, new_platform, new_pump):
                    go_to_chat(serial, new_platform, new_pump)
                else:
                    st.error("Serial Number exists! Use the 'Existing Machine' menu.")
    with col2:
        st.subheader("Troubleshoot Existing Machine")
        machines = get_all_machines()
        if machines:
            options = {f"{m['serial']} | {m['platform']} | {m['pump']}": m for m in machines}
            selected = st.selectbox("Select Machine from Fleet", list(options))
            if st.button("Connect & Troubleshoot"):
                m = options[selected]
                go_to_chat(m["serial"], m["platform"], m["pump"])
        else:
            st.info("No machines registered in the fleet yet.")


def render_case_lists(m, is_gen):
    cases = collect_cases(m["serial"])
    current = st.session_state.case_id
    if not cases:
        st.sidebar.caption("📂 No cases recorded on this machine yet.")
        return
    open_cases = [c for c in cases if not c.get("fix") and c["case_id"] != current]

    if open_cases:
        with st.sidebar.expander(f"🟡 Unsolved cases ({len(open_cases)})",
                                 expanded=not st.session_state.messages):
            st.caption("Continue one: the AI keeps that case's answers and earlier reports, "
                       "and you add what you tried since.")
            for c in reversed(open_cases):
                st.markdown(f"**Case {c['case_id']}** · {local_time(c['started'])}")
                st.caption(short(c["symptom"]))
                if st.button("▶️ Continue this case", key=f"cont_{c['case_id']}",
                             disabled=is_gen, use_container_width=True):
                    resume_case(m, c["case_id"])
                st.divider()

    with st.sidebar.expander(f"📂 Case reports ({len(cases)})"):
        for c in reversed(cases):
            cid = c["case_id"]
            status = "🟢 Closed" if c.get("fix") else "🟡 Open"
            cur = " · *current*" if cid == current else ""
            st.markdown(f"**Case {cid}** · {local_time(c['started'])} · {status}{cur}")
            st.caption(short(c["symptom"]))
            pdf = st.session_state.case_pdf
            if pdf and pdf["case_id"] == cid:
                st.download_button("⬇️ Download case PDF", data=pdf["bytes"],
                                   file_name=pdf["name"], mime="application/pdf",
                                   key=f"dl_{cid}", use_container_width=True)
            elif st.button("📄 Case report", key=f"rep_{cid}", disabled=is_gen,
                           use_container_width=True):
                with st.spinner("Rendering case report..."):
                    export_case(m, cid)
                st.rerun()
            st.divider()


def render_sidebar(m, store):
    st.sidebar.title("🎛️ Diagnostic Tools")
    is_gen = st.session_state.get("is_generating", False)

    st.sidebar.toggle("🌙 Dark mode", key="dark", disabled=is_gen)
    st.sidebar.select_slider("Audience",
                             options=["Intern", "Maintenance Tech", "Senior Engineer"],
                             key="tone", disabled=is_gen)
    st.sidebar.radio("Mode", ["Flash", "Deep"], key="mode", disabled=is_gen, horizontal=True)
    tone, mode = st.session_state.tone, st.session_state.mode
    st.sidebar.caption("This audience gets: " + " → ".join(report_sections(tone)))
    st.sidebar.divider()

    render_case_lists(m, is_gen)
    st.sidebar.divider()

    telemetry = None
    uploaded_log = st.sidebar.file_uploader("📄 Upload Machine Log (.txt)", type=["txt"],
                                            key=f"log_{m['serial']}", disabled=is_gen)
    if uploaded_log:
        telemetry, err = parse_log(uploaded_log.getvalue(), uploaded_log.name)
        if err:
            st.sidebar.error(err)
        if telemetry:
            icons = {"NOMINAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}
            status = telemetry.get("status", "UNKNOWN")
            st.sidebar.markdown(f"**Status:** {icons.get(status, '⚪')} {status}")
            st.sidebar.caption(telemetry.get("drift_summary", ""))
            mismatch = telemetry_mismatch(telemetry, m)
            if mismatch:
                st.sidebar.warning(f"Log may be from another machine: {mismatch}")

    st.sidebar.file_uploader("📷 Upload Defect Image", type=["png", "jpg"], disabled=True)

    history = get_resolutions(m["serial"])
    fleet = get_fleet_resolutions(m["pump"], exclude_serial=m["serial"])
    if history or fleet:
        with st.sidebar.expander(f"🧠 Fleet memory ({len(history)} here, "
                                 f"{len(fleet)} on other {m['pump']} machines)"):
            for r in reversed(history[-5:]):
                st.caption(f"**{local_time(r['timestamp'])[:10]}** · {r['symptom']}\n\n→ {r['fix']}")
            if fleet:
                st.caption("— same pump, other machines —")
                for r in fleet[:5]:
                    st.caption(f"**{r['serial']}** · {r['symptom']}\n\n→ {r['fix']}")

    with st.sidebar.expander("🔎 Retrieval diagnostics"):
        try:
            info = retrieval_report(store)
            st.caption(f"Collection: `{info['collection']}` · chunks: **{info['documents']}**")
            st.caption(f"Distance: **{info['space']}** · cutoff **{info['cutoff']:.3f}** "
                       f"({info['source']})")
            st.caption(f"Hardware-tagged store: **{info['hardware_tagged']}**")
            inv = hardware_inventory(store, m["pump"], m["platform"])
            st.caption(f"Documents about **{m['pump']}**: {len(inv['pump_docs'])} "
                       f"(+{inv['pump_tagged_chunks']} tagged chunks)")
            for d in inv["pump_docs"][:8]:
                st.caption(f"• {d} ({inv['per_doc'].get(d, 0)} chunks)")
            st.caption(f"Documents about **{m['platform']}**: {len(inv['platform_docs'])}")
            for d in inv["platform_docs"][:8]:
                st.caption(f"• {d} ({inv['per_doc'].get(d, 0)} chunks)")
            if not inv["pump_docs"] and not inv["pump_tagged_chunks"]:
                st.warning("No document in the store is about this pump. Check the PDF "
                           "filenames / run `python ingest.py --reset`.")
        except Exception as exc:
            st.caption(f"Unavailable: {exc}")

    st.sidebar.divider()
    if st.sidebar.button("⬅️ Switch Machine", disabled=is_gen):
        st.session_state.current_page = "Landing"
        st.session_state.pdf_bytes = None
        st.session_state.case_pdf = None
        st.rerun()
    return tone, mode, telemetry


def machine_assessment(llm, machine, cases) -> str:
    if not cases:
        return ""
    try:
        parts = []
        for chunk in llm.stream(build_history_messages(machine, cases)):
            parts.append(chunk_text(chunk.content))
        return repair_markdown("".join(parts).strip())
    except Exception as exc:
        return f"### Assessment Summary\nThe automated assessment could not be produced ({exc})."


def render_export_bar(m, llm_report):
    is_gen = st.session_state.get("is_generating", False)
    st.divider()
    st.subheader("Reports")
    st.caption("Individual case reports are in the sidebar under 📂 Case reports.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📚 Machine history report", use_container_width=True, disabled=is_gen):
            with st.spinner("Reviewing the machine's full history..."):
                cases = collect_cases(m["serial"], include_reports=False)
                machine = get_machine(m["serial"]) or m
                assessment = machine_assessment(llm_report, machine, cases)
                st.session_state.pdf_bytes = build_machine_history_pdf(
                    machine, cases, assessment_md=assessment, logo_path=LOGO_PATH)
                st.session_state.pdf_name = f"History_{m['serial']}.pdf"
            st.rerun()
    with c2:
        if st.session_state.pdf_bytes:
            st.download_button(f"⬇️ Download {st.session_state.pdf_name}",
                               data=st.session_state.pdf_bytes,
                               file_name=st.session_state.pdf_name,
                               mime="application/pdf", use_container_width=True,
                               disabled=is_gen)
        else:
            st.caption("Generate the history report to enable the download.")


def render_resolution_bar(m, llm_report):
    st.divider()
    if st.session_state.last_notes:
        with st.expander("⚠️ Quality notes on this report", expanded=False):
            for note in st.session_state.last_notes:
                st.caption(f"• {note}")

    st.subheader("Ticket Resolution")
    res = get_resolution_for_case(m["serial"], st.session_state.case_id)
    subject = urllib.parse.quote(f"Support Request: Machine {m['serial']}")
    body = urllib.parse.quote("Please find attached the exported PDF diagnostic report.")
    mail = f"mailto:support@nswautomations.com.my?subject={subject}&body={body}"

    if res:
        st.success(f"✅ Case closed — {res['fix']}")
        c1, c2 = st.columns(2)
        with c1:
            st.link_button("📧 Draft Email", mail, use_container_width=True)
        with c2:
            if st.button("🔄 New Issue on this Machine", use_container_width=True):
                start_new_case(m, "[SYSTEM] Case closed. Starting a new case on this machine.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("✅ Mark as Fixed", use_container_width=True):
                st.session_state.show_fix_input = True
        with c2:
            if st.button("🔁 Continue troubleshooting this issue", use_container_width=True):
                record(m["serial"], "assistant",
                       "[SYSTEM] Continuing troubleshooting on this case.")
                st.session_state.chat_phase = "AWAIT_UPDATE"
                st.session_state.show_fix_input = False
                st.rerun()
        with c3:
            st.link_button("📧 Draft Email", mail, use_container_width=True)
        with c4:
            if st.button("🔄 New Issue on this Machine", use_container_width=True):
                start_new_case(m)

    if st.session_state.show_fix_input and not res:
        reports = st.session_state.case.get("reports") or []
        causes = extract_causes(reports[-1]) if reports else []
        options = [f"Cause {i}: {name}" for i, (name, _c) in enumerate(causes, 1)]
        options.append("Something else (not in the AI's list)")
        with st.form("resolution_form"):
            choice = st.selectbox("Which cause was it?", options)
            fix_desc = st.text_area("What exactly did you do to fix it?",
                                    placeholder="e.g., Replaced the worn ceramic nozzle (0.2 mm) "
                                                "and re-ran the purge; dot size back in spec.")
            if st.form_submit_button("Save Resolution to Database"):
                if len(fix_desc.strip()) < 8:
                    st.error("Describe the fix in a sentence so the next operator - and the "
                             "AI - can learn from it.")
                else:
                    cause = re.sub(r"^Cause \d+:\s*", "", choice)
                    if choice.startswith("Something else"):
                        cause = "not in the AI's list"
                    save_resolution(m["serial"],
                                    st.session_state.case.get("raw_symptom") or "Unknown",
                                    f"Confirmed cause: {cause}. Fix: {fix_desc.strip()}",
                                    st.session_state.case_id)
                    st.session_state.show_fix_input = False
                    st.rerun()

    render_export_bar(m, llm_report)


# --------------------------------------------------------------------------- #
# Turn handling
# --------------------------------------------------------------------------- #
def run_diagnosis(m, llm_report, tone, results, blueprint, fleet_memory, has_details,
                  continuation=""):
    case = st.session_state.case
    prior = case.get("reports") or []
    instruction = build_diagnosis_instruction(m["pump"], has_details, tone, results=results,
                                              symptom=case.get("raw_symptom", ""))
    if continuation:
        instruction = f"{continuation}\n{instruction}"
        llm_report = make_llm("report", retry_seed())   # fresh sampling for a new round

    main = st.empty()
    status = st.empty()
    scratch = st.container()
    state = {"first": True}

    def stream_fn(messages):
        if state["first"]:
            state["first"] = False
            return make_stream_fn(llm_report, main)(messages)
        with scratch:
            return make_stream_fn(llm_report, st.empty())(messages)

    def attempt(instr):
        state["first"] = True
        msgs = build_llm_messages(tone, m, results, case, instr, blueprint, fleet_memory)
        return generate_report(stream_fn, msgs, m["pump"], notify=lambda s: status.info(s),
                               tone=tone, allowed_sources=source_names(results))

    text, problems, notes = attempt(instruction)

    # NEW: if a continued case repeats earlier causes, regenerate once with them forbidden.
    again = repeated_causes(text, prior) if continuation else []
    if again:
        status.info(f"{len(again)} cause(s) repeat an earlier round - asking for new ones...")
        banned = "\n".join(f"  - {c}" for c in again)
        retry_instr = (f"{instruction}\n\n=== FORBIDDEN CAUSES (already tried, did NOT fix "
                       f"the defect) ===\n{banned}\nDo NOT list any of these again, even "
                       f"reworded. Replace them with physically different mechanisms.")
        t2, p2, n2 = attempt(retry_instr)
        if len(repeated_causes(t2, prior)) < len(again) and len(p2) <= len(problems):
            text, problems, notes = t2, p2, n2
        again = repeated_causes(text, prior)

    status.empty()
    scratch.empty()
    text = repair_markdown(text)
    main.markdown(text)
    if problems:
        st.warning("The model did not fully follow the report layout "
                   f"({'; '.join(problems)}). Treat this output with caution.")
    if again:
        notes = [f"{len(again)} cause(s) repeat an earlier round: {', '.join(again)[:160]}"
                 " - check whether the update really leaves them open"] + list(notes)
    st.session_state.last_notes = notes
    return text


def finish_report(m, llm_report, tone, results, blueprint, fleet_memory, has_details,
                  continuation=""):
    case = st.session_state.case
    response = run_diagnosis(m, llm_report, tone, results, blueprint, fleet_memory,
                             has_details, continuation)
    record(m["serial"], "assistant", response)
    if not is_out_of_scope(response):
        case["reports"].append(response)
        case["used_keys"] = sorted(set(case["used_keys"])
                                   | {chunk_key(d) for d, _ in results}
                                   | cited_pages(response))
        st.session_state.chat_phase = "LOCKED"


def answer_question(m, llm_report, tone, results, blueprint, question):
    instruction = build_question_instruction(m["pump"], m["platform"], tone, results)
    temp_case = {"symptom": f"(no defect reported) Operator question: {question}"}
    msgs = build_llm_messages(tone, m, results, temp_case, instruction, blueprint)
    ph = st.empty()
    text = make_stream_fn(llm_report, ph)(msgs)
    text, _ = fix_citations(text, source_names(results))
    idx = text.find(ANSWER_HEADING)
    text = text[idx:] if idx > 0 else text if idx == 0 else f"{ANSWER_HEADING}\n{text}"
    text = repair_markdown(text)
    ph.markdown(text)
    record(m["serial"], "assistant", text)


def handle_turn(m, llms, tone, diag_mode, user_text, telemetry, blueprint):
    store, llm_ask, llm_report, llm_route = llms
    case = st.session_state.case
    if telemetry:
        case["telemetry_note"] = telemetry_line(telemetry)
        case["telemetry_digest"] = telemetry_digest(telemetry, m)
    phase = st.session_state.chat_phase
    results = []

    if phase in ("AWAIT_SYMPTOM", "AWAIT_UPDATE"):
        route = route_message(llm_route, store, user_text, m["pump"], m["platform"],
                              followup=(phase == "AWAIT_UPDATE"))
        if route.label == "OFF_TOPIC":
            reply = off_topic_reply(route.topic, m["platform"], m["pump"])
            st.markdown(reply)
            if route.method == "embedding" and route.best is not None:
                st.caption(f"(Router unavailable - decided by document distance "
                           f"{route.best:.3f} vs cutoff {route.cutoff:.3f}.)")
            record(m["serial"], "assistant", reply)
            return
        if route.label == "GREETING":
            reply = greeting_reply(m["serial"], m["platform"], m["pump"])
            st.markdown(reply)
            record(m["serial"], "assistant", reply)
            return
        if route.label == "QUESTION":
            answer_question(m, llm_report, tone, route.results, blueprint, user_text)
            return
        results = route.results

    if phase == "AWAIT_SYMPTOM":
        case["raw_symptom"] = user_text
        note = case.get("telemetry_note")
        case["symptom"] = user_text + (f"\nTelemetry: {note}" if note else "")
        case["cached"] = results
        case["memory"] = None
        fleet_memory = case_memory(m, store, case)

        if diag_mode == "Flash":
            finish_report(m, llm_report, tone, results, blueprint, fleet_memory,
                          has_details=bool(note))
        else:
            instruction = build_inquiry_instruction(m["pump"], 1, tone)
            msgs = build_llm_messages(tone, m, results, case, instruction, blueprint,
                                      fleet_memory, kind="ask")
            response = stream_inquiry(llm_ask, msgs, st.empty(), 1)
            record(m["serial"], "assistant", response)
            if not is_out_of_scope(response):
                case["pending_q"] = response
                case["rounds"] = 1
                st.session_state.chat_phase = "AWAIT_ANSWERS"

    elif phase == "AWAIT_ANSWERS":
        fleet_memory = case_memory(m, store, case)
        skipped = is_non_answer(user_text)
        answer = "Operator does not know / gave no further details." if skipped else user_text
        case["transcript"].append((case["pending_q"], answer))

        details = [a for _, a in case["transcript"]
                   if not is_non_answer(a) and not a.startswith("Operator does not know")]
        query = " ".join([case["symptom"]] + details)[:800]
        results = retrieve_relevant(store, query, m["pump"], m["platform"]) or case["cached"]

        if details and case.get("rounds", 0) < MAX_INQUIRY_ROUNDS and not skipped:
            round_no = case["rounds"] + 1
            instruction = build_inquiry_instruction(m["pump"], round_no, tone)
            msgs = build_llm_messages(tone, m, results, case, instruction, blueprint,
                                      fleet_memory, kind="ask")
            response = stream_inquiry(llm_ask, msgs, st.empty(), round_no)
            record(m["serial"], "assistant", response)
            if not is_out_of_scope(response):
                case["pending_q"] = response
                case["rounds"] = round_no
                case["cached"] = results
                return

        finish_report(m, llm_report, tone, results, blueprint, fleet_memory,
                      has_details=bool(details) or bool(case.get("telemetry_note")))

    elif phase == "AWAIT_UPDATE":
        fleet_memory = case_memory(m, store, case)
        case["updates"].append(user_text)
        query = " ".join([case.get("raw_symptom", ""), user_text])[:800]
        results = retrieve_relevant(store, query, m["pump"], m["platform"],
                                    exclude=set(case.get("used_keys") or []))
        block = build_continuation_block(case.get("reports") or [],
                                         len(case.get("reports") or []) + 1)
        finish_report(m, llm_report, tone, results, blueprint, fleet_memory,
                      has_details=True, continuation=block)


def render_chat_page():
    m = st.session_state.active_machine
    if not m:
        st.session_state.current_page = "Landing"
        st.rerun()

    try:
        llms = load_ai_backend()
    except SystemExit as exc:
        st.error(f"Backend initialisation failed: {exc}")
        st.stop()
    store = llms[0]

    blueprint, bp_warning = load_blueprint(m["platform"], m["pump"])
    tone, diag_mode, telemetry = render_sidebar(m, store)
    if bp_warning:
        st.sidebar.warning(bp_warning)

    phase = st.session_state.chat_phase
    st.title(f"🔬 Diagnostics: {m['serial']}")
    st.caption(f"Platform: {m['platform']} | Pump: {m['pump']} | Case "
               f"{st.session_state.case_id} ({PHASE_LABEL.get(phase, phase)}) | "
               f"{MIN_CAUSE_ROWS}-{MAX_CAUSE_ROWS} causes, as many as the evidence supports")

    if not st.session_state.messages:
        open_n = sum(1 for c in collect_cases(m["serial"])
                     if not c.get("fix") and c["case_id"] != st.session_state.case_id)
        if open_n:
            st.info(f"This machine has **{open_n} unsolved case(s)**. Continue one from "
                    f"**🟡 Unsolved cases** in the sidebar, or describe a new issue below.")

    for msg in st.session_state.messages:
        content = msg.get("content") or ""
        if content.startswith("[SYSTEM]"):
            st.caption(f"ℹ️ {content[8:].strip()}")
            continue
        with st.chat_message(msg["role"]):
            st.markdown(content)

    if phase == "LOCKED":
        render_resolution_bar(m, llms[2])
        return

    is_gen = st.session_state.get("is_generating", False)
    placeholder = {
        "AWAIT_SYMPTOM": "Describe the defect (what you see, where, since when) or ask a "
                         "question about this machine...",
        "AWAIT_ANSWERS": "Answer the questions above (numbered answers are fine, 'idk' to skip)...",
        "AWAIT_UPDATE": "What did you try since the last report, and what happened? "
                        "e.g. 'Replaced the nozzle, dots still 20% too large'...",
    }.get(phase, "")
    if phase == "AWAIT_UPDATE":
        st.info("🔁 Continuing this case. Tell me what you tried and what you measured - the "
                "next report will rule those out and work from new evidence.")

    cols = st.columns([0.05, 0.95], gap="small") if speech_to_text else [None, st.container()]
    if speech_to_text:
        with cols[0]:
            mic_result = speech_to_text(start_prompt="🎙️", stop_prompt="🔴 Rec", language="en",
                                        use_container_width=True, just_once=True, key="STT")
            if mic_result and not is_gen:
                st.session_state.draft_text = mic_result
                st.rerun()

    with cols[1]:
        with st.form("chat_input_form", clear_on_submit=True):
            user_text = st.text_area("Message", value=st.session_state.draft_text,
                                     placeholder=placeholder, label_visibility="collapsed",
                                     height=100, disabled=is_gen)
            submit_btn = st.form_submit_button("Send (Ctrl+Enter)", disabled=is_gen)

    render_export_bar(m, llms[2])

    if submit_btn and user_text and user_text.strip():
        st.session_state.pending_query = user_text.strip()[:MAX_INPUT_CHARS]
        st.session_state.draft_text = ""
        st.session_state.is_generating = True
        st.rerun()

    if st.session_state.get("is_generating") and st.session_state.get("pending_query"):
        display_text = st.session_state.pending_query
        note = telemetry_line(telemetry)
        # BUG FIX: telemetry used to be appended to EVERY message, including "1. epoxy 2. yes".
        if note and phase == "AWAIT_SYMPTOM":
            display_text += f"\n\n> 📊 **Telemetry Injected:** {note}"

        record(m["serial"], "user", display_text)
        with st.chat_message("user"):
            st.markdown(display_text)

        with st.chat_message("assistant"):
            try:
                handle_turn(m, llms, tone, diag_mode, st.session_state.pending_query,
                            telemetry, blueprint)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                st.caption(f"Is Ollama running, and is '{LLM_MODEL}' pulled? "
                           f"Try: ollama pull {LLM_MODEL}")
            finally:
                st.session_state.is_generating = False
                st.session_state.pending_query = ""
                st.rerun()


if st.session_state.dark:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
if st.session_state.current_page == "Landing":
    render_landing_page()
else:
    render_chat_page()
