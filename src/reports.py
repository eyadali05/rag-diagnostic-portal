"""NSW Defect Detective - the two report types.

This module sits on top of report_pdf.py and uses its Markdown -> ReportLab engine, fonts,
palette and page furniture. Nothing in the rendering layer changed; what changed is what
goes into the story.

    build_case_report_pdf(...)     one troubleshooting case, as a report
    build_machine_history_pdf(...) the whole machine, with an AI assessment

WHY TWO TYPES
-------------
The old build_report_pdf() dumped the machine's entire chat log as a transcript, with
"OPERATOR" / "AI DEFECT DETECTIVE" labels, every "[SYSTEM] Session reopened" marker and
every SECURITY_ABORT. It read like a log file, and because case_start_id was computed but
never used, a machine with six months of history exported all six months every time.

  * The CASE report is scoped to one case_id: the operator's inputs as a facts block, then
    the cause table, then the troubleshooting steps, then the confirmed fix if the case was
    closed. No chat framing.
  * The HISTORY report is the maintenance planner's document: every case in date order, with
    its top causes and whether it was ever closed, preceded by an AI assessment of the
    machine as a whole.

The disclaimer is no longer repeated in the footer of every page (footer_note=""); it
appears once, under the header block. The page number stays - a report that ends up on
someone's desk detached from its email needs one.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import re
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, Spacer, Table, TableStyle

from report_pdf import (C, NSW_NAVY, NSW_ORANGE, NSW_RULE, PALETTE, _Doc, _styles,
                        clean_text, md_to_flowables, repair_markdown)

DISCLAIMER = ("AI-generated diagnostic support. Verify every step against the current NSW "
              "Automation manuals before acting on it.")


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #
def _facts_table(rows, st, avail):
    """The 'here is what was entered' block. Two columns, label left, value right."""
    data = [[Paragraph(f"<b>{_html.escape(str(k))}</b>", st["cell"]),
             Paragraph(_html.escape(clean_text(str(v))).replace("\n", "<br/>"), st["cell"])]
            for k, v in rows if v not in (None, "")]
    if not data:
        return None
    t = Table(data, colWidths=[avail * 0.24, avail * 0.76], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), C("zebra")),
        ("GRID", (0, 0), (-1, -1), 0.4, NSW_RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _header_block(serial, subtitle, title_text, st):
    return [
        Paragraph(f"{_html.escape(title_text)} &nbsp;"
                  f"<font color='{PALETTE['orange']}'>{_html.escape(serial)}</font>",
                  st["title"]),
        Paragraph(_html.escape(subtitle or ""), st["subtitle"]),
        Paragraph(f"<i>{_html.escape(DISCLAIMER)}</i>", st["src"]),
        HRFlowable(width="100%", thickness=1.4, color=NSW_NAVY, spaceBefore=6, spaceAfter=10),
    ]


def _sources_block(sources, st):
    if not sources:
        return []
    out = [Spacer(1, 4),
           HRFlowable(width="100%", thickness=0.8, color=NSW_RULE, spaceAfter=6),
           Paragraph("Cited sources", st["h2"])]
    for n, src in enumerate(sources, 1):
        out.append(Paragraph(
            f'<super><font color="{PALETTE["orange"]}">{n}</font></super> '
            f'&nbsp;{_html.escape(src)}', st["src"]))
    return out


def _build(doc, story, fallback_text=""):
    """Build, with the same never-crash fallback the old exporter had."""
    try:
        doc.build(story)
    except Exception:  # noqa: BLE001
        buf = BytesIO()
        alt = _Doc(buf, doc.serial, doc.subtitle, doc.logo_path, footer_note="")
        st = _styles()
        alt.build([Paragraph(_html.escape(clean_text(fallback_text)).replace("\n", "<br/>"),
                             st["body"])] or [Paragraph("No content.", st["body"])])
        return buf.getvalue()
    return None


# --------------------------------------------------------------------------- #
# Case extraction
# --------------------------------------------------------------------------- #
DIAGNOSIS_MARKERS = ("### Root Cause Analysis", "| # |", "| Cause")


def is_diagnosis(content: str) -> bool:
    return any(m in (content or "") for m in DIAGNOSIS_MARKERS)


# Assistant replies that are NOT part of a troubleshooting case: rejections, greetings and
# documentation answers. The operator message that triggered one is dropped with it.
NON_CASE_PREFIXES = ("SECURITY_ABORT", "OUT_OF_SCOPE", "\U0001F6AB", "\U0001F44B",
                     "### Answer")


def is_non_case_reply(content: str) -> bool:
    return (content or "").strip().startswith(NON_CASE_PREFIXES)


def _case_messages(messages):
    """BUG FIX: an off-topic prompt ("recipe for koshary") that was rejected stayed in the
    case as its first operator input, so it became the case's 'Reported Symptom' in the PDF
    and in the history report. Drop rejected/greeting/Q&A exchanges entirely."""
    msgs = [m for m in (messages or []) if (m.get("content") or "").strip()
            and not (m.get("content") or "").strip().startswith("[SYSTEM]")]
    out = []
    for msg in msgs:
        role = (msg.get("role") or "assistant").lower()
        if role != "user" and is_non_case_reply(msg.get("content")):
            if out and (out[-1].get("role") or "").lower() == "user":
                out.pop()
            continue
        out.append(msg)
    return out


def split_case_full(messages) -> dict:
    """symptom inputs, inquiry rounds, every report, and operator updates between reports."""
    inputs, rounds, reports, updates = [], [], [], []
    pending_q = None
    for msg in _case_messages(messages):
        role = (msg.get("role") or "assistant").lower()
        content = (msg.get("content") or "").strip()
        if role == "user":
            if pending_q:
                rounds.append((pending_q, content))
                pending_q = None
            elif reports:
                updates.append((len(reports), content))
            else:
                inputs.append(content)
        elif is_diagnosis(content):
            reports.append(content)
            pending_q = None
        elif "?" in content:
            pending_q = content
    return {"inputs": inputs, "rounds": rounds, "reports": reports, "updates": updates,
            "pending_q": pending_q}


def split_case(messages):
    """(operator_inputs, inquiry rounds, latest report markdown) for one case."""
    full = split_case_full(messages)
    return full["inputs"], full["rounds"], (full["reports"][-1] if full["reports"] else "")


def case_summary(messages, max_causes: int = 4):
    """(symptom, [cause names]) for one case - used by the history report and the AI prompt."""
    inputs, _rounds, diagnosis = split_case(messages)
    symptom = inputs[0] if inputs else ""
    # BUG FIX: the stored line is "> 📊 **Telemetry Injected:**" - the emoji between ">" and
    # "**" meant this never matched, so the telemetry blob leaked into every case symptom.
    symptom = re.sub(r"\n*>\s*(?:\S+\s*)?\*\*Telemetry Injected.*", "", symptom,
                     flags=re.S).strip()
    causes = []
    if diagnosis:
        for line in repair_markdown(diagnosis).splitlines():
            m = re.match(r"^####\s+Cause\s*\d+\s*[:.-]?\s*(.+)$", line.strip())
            if m:
                causes.append(m.group(1).strip())
        if not causes:
            for line in diagnosis.splitlines():
                if line.strip().startswith("|") and line.count("|") >= 5:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cells) > 1 and not set(cells[0]) <= set("-: "):
                        if cells[1].lower() not in {"cause", ""}:
                            causes.append(cells[1])
    return symptom, causes[:max_causes]


# --------------------------------------------------------------------------- #
# 1. Case report
# --------------------------------------------------------------------------- #
def build_case_report_pdf(serial: str, messages, platform=None, pump=None, tone=None,
                          mode=None, logo_path=None, case_id=None, resolution=None,
                          telemetry_note=None, generated_at=None) -> bytes:
    """One troubleshooting case as a report.

    `messages`   - the chat for this case only (database.get_chat_history(serial, case_id=...))
    `resolution` - database.get_resolution_for_case(...) or None
    """
    st = _styles()
    stamp = (generated_at or _dt.datetime.now()).strftime("%Y-%m-%d %H:%M")
    subtitle = " | ".join(str(x) for x in [platform, pump,
                                           f"Case {case_id}" if case_id is not None else None,
                                           stamp] if x)
    buf = BytesIO()
    doc = _Doc(buf, serial, subtitle, logo_path, footer_note="",
               doc_title=f"NSW Case Report {serial}")
    avail = doc.width
    sources: list = []

    full = split_case_full(messages)
    inputs, rounds = full["inputs"], full["rounds"]
    diagnosis = full["reports"][-1] if full["reports"] else ""
    story = _header_block(serial, subtitle, "Diagnostic Case Report", st)

    # --- operator inputs, as facts rather than as a chat transcript ---------- #
    story.append(Paragraph("Case Details", st["h1"]))
    facts = [("Serial number", serial), ("Platform", platform), ("Pump unit", pump),
             ("Audience", tone), ("Diagnostic mode", mode), ("Generated", stamp)]
    if resolution:
        facts.append(("Status", f"CLOSED - fix confirmed {str(resolution.get('timestamp'))[:10]}"))
    else:
        facts.append(("Status", "OPEN - no confirmed fix recorded"))
    t = _facts_table(facts, st, avail)
    if t:
        story += [t, Spacer(1, 8)]

    story.append(Paragraph("Reported Symptom", st["h2"]))
    symptom = inputs[0] if inputs else "(not recorded)"
    story += md_to_flowables(symptom, st, avail, sources)
    if telemetry_note:
        story.append(Paragraph(f"<b>Telemetry:</b> {_html.escape(clean_text(telemetry_note))}",
                               st["quote"]))
    for extra in inputs[1:]:
        story += md_to_flowables(extra, st, avail, sources)

    if rounds:
        story.append(Paragraph("Operator Responses", st["h2"]))
        qa = []
        for q, a in rounds:
            qa.append((re.sub(r"\s+", " ", clean_text(q))[:400], clean_text(a)))
        t = _facts_table(qa, st, avail)
        if t:
            story += [t, Spacer(1, 8)]

    # --- continued cases: what each earlier round proposed and what came back -- #
    if len(full["reports"]) > 1 or full["updates"]:
        story.append(Paragraph("Troubleshooting Rounds", st["h2"]))
        rows = []
        for n, rep_md in enumerate(full["reports"], 1):
            _s, causes = case_summary([{"role": "user", "content": "x"},
                                       {"role": "assistant", "content": rep_md}], 6)
            label = "Latest round" if n == len(full["reports"]) else f"Round {n}"
            rows.append((f"{label} - causes", "\n".join(f"- {c}" for c in causes) or "-"))
            for after, upd in full["updates"]:
                if after == n:
                    rows.append((f"Operator update after round {n}", upd))
        t = _facts_table(rows, st, avail)
        if t:
            story += [t, Spacer(1, 8)]

    # --- the report itself: table then steps, nothing else ------------------ #
    if diagnosis:
        story += md_to_flowables(diagnosis, st, avail, sources)
    else:
        story.append(Paragraph("No diagnostic report was produced for this case.", st["body"]))

    # --- confirmed fix ------------------------------------------------------ #
    if resolution and resolution.get("fix"):
        story.append(Spacer(1, 6))
        box = Table([[Paragraph(
            f"<b>Confirmed fix</b> &nbsp;<font size=7.5 color='{PALETTE['grey']}'>"
            f"{_html.escape(str(resolution.get('timestamp') or '')[:16])}</font><br/>"
            f"{_html.escape(clean_text(resolution['fix']))}", st["body"])]],
            colWidths=[avail])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C("cream")),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, C("high")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(KeepTogether([box]))

    story += _sources_block(sources, st)
    fallback = _build(doc, story, diagnosis or symptom)
    return fallback if fallback is not None else buf.getvalue()


# --------------------------------------------------------------------------- #
# 2. Machine history report
# --------------------------------------------------------------------------- #
def build_machine_history_pdf(machine: dict, cases: list, assessment_md: str = "",
                              logo_path=None, generated_at=None) -> bytes:
    """The machine's life story, with an AI assessment at the front.

    `cases` is a list of dicts:
        {case_id, started, symptom, causes: [str], fix: str|None, fixed_at: str|None,
         diagnosis: str (markdown, optional - included when include_reports is on)}
    """
    st = _styles()
    serial = machine.get("serial", "?")
    stamp = (generated_at or _dt.datetime.now()).strftime("%Y-%m-%d %H:%M")
    subtitle = " | ".join(str(x) for x in [machine.get("platform"), machine.get("pump"),
                                           f"{len(cases)} cases", stamp] if x)
    buf = BytesIO()
    doc = _Doc(buf, serial, subtitle, logo_path, footer_note="",
               doc_title=f"NSW Machine History {serial}")
    avail = doc.width
    sources: list = []

    story = _header_block(serial, subtitle, "Machine History", st)

    closed = sum(1 for c in cases if c.get("fix"))
    t = _facts_table([
        ("Serial number", serial),
        ("Platform", machine.get("platform")),
        ("Pump unit", machine.get("pump")),
        ("In fleet since", str(machine.get("date_added") or "")[:10]),
        ("Cases on record", f"{len(cases)} total - {closed} closed, {len(cases) - closed} open"),
        ("First case", str(cases[0].get("started") or "")[:10] if cases else "-"),
        ("Most recent", str(cases[-1].get("started") or "")[:10] if cases else "-"),
    ], st, avail)
    if t:
        story += [t, Spacer(1, 10)]

    if assessment_md:
        story += md_to_flowables(assessment_md, st, avail, sources)
        story.append(Spacer(1, 6))

    story.append(Paragraph("Case Log", st["h1"]))
    if not cases:
        story.append(Paragraph("No troubleshooting cases recorded for this machine.",
                               st["body"]))
    for c in cases:
        when = str(c.get("started") or "")[:10]
        status = "CLOSED" if c.get("fix") else "OPEN"
        colour = PALETTE["high"] if c.get("fix") else PALETTE["low"]
        head = Paragraph(
            f"Case {c.get('case_id')} &nbsp;<font size=8 color='{PALETTE['grey']}'>{when}</font>"
            f" &nbsp;<font size=8 color='{colour}'><b>{status}</b></font>", st["h2"])
        rows = [("Symptom", c.get("symptom") or "(not recorded)")]
        if c.get("causes"):
            rows.append(("Causes considered",
                         "\n".join(f"{i}. {x}" for i, x in enumerate(c["causes"], 1))))
        rows.append(("Outcome", c.get("fix") or "No confirmed fix recorded."))
        body = _facts_table(rows, st, avail)
        story.append(KeepTogether([head, body] if body else [head]))
        if c.get("diagnosis"):
            story += md_to_flowables(c["diagnosis"], st, avail, sources)
        story.append(Spacer(1, 8))

    story += _sources_block(sources, st)
    fallback = _build(doc, story, assessment_md)
    return fallback if fallback is not None else buf.getvalue()
