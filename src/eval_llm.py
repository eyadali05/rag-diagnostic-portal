#!/usr/bin/env python3
"""NSW Defect Detective - headless accuracy checks (no Streamlit needed).

    python eval_llm.py --quick      # router + retrieval only            (~1-2 min)
    python eval_llm.py              # + full reports, tones, continuation, fleet memory
    python eval_llm.py --pump "PistonPRO 4+" --platform "i-DR S420"

Every check prints PASS / FAIL / WARN with the evidence, then a scoreboard. Run it once
before the demo: a FAIL tells you exactly which behaviour to avoid showing live.
"""
import argparse
import re
import sys
import time

from model import (build_continuation_block, build_diagnosis_instruction, build_fleet_memory,
                   build_inquiry_instruction, build_messages, cause_rows, chunk_key,
                   cited_pages, classify_intent, extract_causes, generate_report,
                   get_hardware_blueprint, hardware_inventory, load_prompt_template,
                   make_llm, open_vector_store, renumber_questions, report_sections,
                   retrieve_relevant, source_names, stream_reply)

RESULTS = []


def check(name, ok, evidence="", warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((tag, name))
    print(f"  [{tag}] {name}" + (f"\n         {evidence}" if evidence else ""))


ROUTER_CASES = [
    ("give me a recipe for koshary", "OFF_TOPIC"),
    ("what's the weather in Penang tomorrow", "OFF_TOPIC"),
    ("write python code to sort a list", "OFF_TOPIC"),
    ("hello", "GREETING"),
    ("the machine is dispensing more fluid than required", "DEFECT"),
    ("dots missing on the right side of the board since we changed syringe", "DEFECT"),
    ("air bubbls in the glue", "DEFECT"),
    ("how do I purge air from the syringe on this pump?", "QUESTION"),
    ("what is the max fluid pressure this pump supports?", "QUESTION"),
    ("koshary recipe pls, also my dots are too big", "DEFECT"),
]

SCENARIOS = [
    # (symptom, expected class keyword)
    ("the machine is dispensing more fluid than required, dots are too big", "Excess"),
    ("dots are too small and getting smaller over the shift", "Insufficient"),
    ("there are air bubbles inside the dispensed dots", "bubble"),
]


def section_count(text, name):
    return len(re.findall(rf"(?m)^###\s+{re.escape(name)}\s*$", text))


def run_report(llm, template, tone, platform, pump, blueprint, results, case_file,
               continuation="", quiet=True):
    instr = build_diagnosis_instruction(pump, True, tone, results=results)
    if continuation:
        instr = continuation + "\n" + instr
    msgs = build_messages(template, tone, platform, pump,
                          {"blueprint": blueprint, "chunks": results}, case_file, instr)

    def stream_fn(m):
        if quiet:
            return "".join(str(c.content) for c in llm.stream(m)).strip()
        return stream_reply(llm, m)

    t0 = time.time()
    text, problems, notes = generate_report(stream_fn, msgs, pump, tone=tone,
                                            allowed_sources=source_names(results))
    return text, problems, notes, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pump", default="PistonPRO 4+")
    ap.add_argument("--platform", default="i-DR S420")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    pump, platform = a.pump, a.platform

    store = open_vector_store()
    template = load_prompt_template()
    router, ask, report = make_llm("route"), make_llm("ask"), make_llm("report")

    print("\n=== 1. ROUTER (off-topic / question / greeting / defect) ===")
    for text, want in ROUTER_CASES:
        got, topic = classify_intent(router, text, pump, platform)
        check(f"'{text}' -> {want}", got == want, f"got {got} ({topic})")

    print("\n=== 2. RETRIEVAL COVERAGE (are this machine's documents used?) ===")
    inv = hardware_inventory(store, pump, platform)
    check(f"store has documents about {pump}", bool(inv["pump_docs"] or
                                                     inv["pump_tagged_chunks"]),
          f"docs={inv['pump_docs']} tagged_chunks={inv['pump_tagged_chunks']}")
    check(f"store has documents about {platform}", bool(inv["platform_docs"]),
          f"docs={inv['platform_docs']}", warn=True)
    hw_docs = set(inv["pump_docs"]) | set(inv["platform_docs"])
    for symptom, _ in SCENARIOS:
        res = retrieve_relevant(store, symptom, pump, platform)
        names = [chunk_key(d) for d, _ in res]
        tagged = [n for (d, _), n in zip(res, names)
                  if n.split("|")[0] in hw_docs or pump in str((d.metadata or {}).get("pumps"))]
        check(f"retrieval for '{symptom[:40]}' includes machine-specific pages",
              bool(tagged) or not (hw_docs or inv["pump_tagged_chunks"]),
              f"{len(res)} chunks: {names}")
        check("retrieval draws on more than one document",
              len({n.split('|')[0] for n in names}) > 1, f"{names}", warn=True)
        excl = retrieve_relevant(store, symptom, pump, platform, exclude=set(names))
        check("exclusion returns NEW pages (used when continuing a case)",
              not (set(chunk_key(d) for d, _ in excl) & set(names)) or not excl,
              f"{[chunk_key(d) for d, _ in excl]}")

    if a.quick:
        return scoreboard()

    blueprint, bp_warn = get_hardware_blueprint(store, platform, pump)
    check("hardware blueprint found", not bp_warn, bp_warn or "", warn=True)

    print("\n=== 3. INQUIRY NUMBERING (round 2 must start at 1) ===")
    res = retrieve_relevant(store, SCENARIOS[0][0], pump, platform)
    case = (f"=== CASE FILE ===\nReported symptom: {SCENARIOS[0][0]}\n\nFollow-up round 1, "
            f"questions asked:\n1. Material?\n2. Too much or too little?\n3. Continuous?\n"
            f"Operator's answer: 1. epoxy 2. too much 3. continuous")
    msgs = build_messages(template, "Maintenance Tech", platform, pump,
                          {"blueprint": blueprint, "chunks": res}, case,
                          build_inquiry_instruction(pump, 2, "Maintenance Tech"))
    raw = "".join(str(c.content) for c in ask.stream(msgs))
    fixed = renumber_questions(raw)
    first = re.search(r"(?m)^\s*(\d+)[.)\-]", raw)
    check("model's own round-2 numbering starts at 1", bool(first) and first.group(1) == "1",
          f"raw first number: {first.group(1) if first else None}", warn=True)
    check("displayed round-2 numbering starts at 1 (after renumbering)",
          fixed.lstrip().startswith("1."), fixed.splitlines()[0][:80] if fixed else "")

    print("\n=== 4. REPORTS: accuracy + format per scenario (Maintenance Tech) ===")
    first_report = None
    for symptom, want in SCENARIOS:
        res = retrieve_relevant(store, symptom, pump, platform)
        text, problems, notes, secs = run_report(
            report, template, "Maintenance Tech", platform, pump, blueprint, res,
            f"=== CASE FILE ===\nReported symptom: {symptom}", quiet=not a.verbose)
        first_report = first_report or (symptom, res, text)
        rows = cause_rows(text)
        cls = re.search(r"Class:\s*(.+)", text)
        highs = sum(1 for r in rows if r and r[-1].lower().startswith("high"))
        cited = {c.split("|")[0] for c in cited_pages(text)}
        print(f"  -- '{symptom[:50]}' ({secs:.0f}s)")
        check("layout valid (no structural problems)", not problems, "; ".join(problems))
        check(f"defect class matches ('{want}')", bool(cls) and want.lower() in
              cls.group(1).lower(), cls.group(1) if cls else "no Class line")
        check(f"{len(rows)} causes (2-5)", 2 <= len(rows) <= 5)
        check("at most one High", highs <= 1, f"{highs} High")
        check("all citations were actually retrieved", cited <= source_names(res),
              f"cited={sorted(cited)}")
        check("cites at least one machine-specific document", bool(cited & hw_docs),
              f"cited={sorted(cited)} hw={sorted(hw_docs)[:4]}", warn=True)
        check("no section duplicated", all(section_count(text, s) <= 1 for s in
                                           report_sections("Maintenance Tech")))

    print("\n=== 5. TONES: one Deep Technical Analysis, different depth per audience ===")
    symptom = SCENARIOS[0][0]
    res = retrieve_relevant(store, symptom, pump, platform)
    lengths = {}
    for tone in ("Intern", "Senior Engineer"):
        text, problems, _, _ = run_report(report, template, tone, platform, pump, blueprint,
                                          res, f"=== CASE FILE ===\nReported symptom: {symptom}")
        lengths[tone] = text
        if tone == "Senior Engineer":
            n = section_count(text, "Deep Technical Analysis")
            check("Senior: exactly one 'Deep Technical Analysis'", n == 1, f"found {n}")
            check("Senior: uses the 4 labelled bullets",
                  all(k in text for k in ("Governing physics", "Decisive experiment")),
                  warn=True)
        else:
            check("Intern: has Escalation Protocol", "Escalation Protocol" in text)
            check("Intern: explains terms in brackets", text.count("(") >= 4, warn=True)

    print("\n=== 6. CONTINUATION: next round must not repeat ruled-out causes ===")
    symptom, res, text = first_report
    causes = extract_causes(text)
    if causes:
        top = causes[0][0]
        update = f"I tried cause 1 ({top}) - adjusted/replaced it, no change, same defect."
        used = {chunk_key(d) for d, _ in res} | cited_pages(text)
        res2 = retrieve_relevant(store, symptom + " " + update, pump, platform, exclude=used)
        cf = (f"=== CASE FILE ===\nReported symptom: {symptom}\n\n=== OPERATOR UPDATES ===\n"
              f"Update 1: {update}")
        text2, problems2, _, _ = run_report(make_llm("report", 7), template,
                                            "Maintenance Tech", platform, pump, blueprint,
                                            res2, cf, build_continuation_block([text], 2))
        names2 = [n.lower() for n, _ in extract_causes(text2)]
        check("ruled-out cause not listed again",
              not any(top.lower()[:25] in n for n in names2), f"round 2: {names2}")
        check("round 2 cites different pages", not (cited_pages(text2) & cited_pages(text))
              or not cited_pages(text2), f"{sorted(cited_pages(text2))}", warn=True)
        check("round 2 layout valid", not problems2, "; ".join(problems2))

    print("\n=== 7. FLEET MEMORY: a confirmed fix on this machine should be used ===")
    memory = build_fleet_memory(
        symptom, [{"case_id": 0, "started": "2026-09-01", "symptom": symptom,
                   "causes": [], "fix": "Confirmed cause: worn piston seal. Fix: replaced "
                                        "the piston seal kit, dot size back in spec."}],
        [], None, pump)
    cf = f"=== CASE FILE ===\nReported symptom: {symptom}\n{memory}"
    text3, _, _, _ = run_report(report, template, "Maintenance Tech", platform, pump,
                                blueprint, res, cf)
    check("report uses the machine's confirmed past fix",
          bool(re.search(r"seal|fleet history", text3, re.I)),
          "looked for 'seal' / 'Fleet history' in the report")
    return scoreboard()


def scoreboard():
    n = {k: sum(1 for t, _ in RESULTS if t == k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\n=== SCORE: {n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail ===")
    for t, name in RESULTS:
        if t == "FAIL":
            print(f"  FAIL: {name}")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
