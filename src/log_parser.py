"""NSW machine-log parser.

Line format written by the nFlow log (and generate_machine_logs.py):
    TIMESTAMP | EQUIPMENT | CATEGORY | LEVEL | EVENT | detail [| cycle data]

BUG FIXES vs the previous version:
  * Alarms were taken as `line.split("|")[-1]`, i.e. only the LAST field - the cycle data.
    The event name (MISSING_DOT, AIR_PRESSURE_DIP, ...) and the level were thrown away, so
    the AI never saw what actually fired.
  * Because the level was thrown away, `"ERROR" in alarm` could never be true: status was
    never CRITICAL, even when the log ended in PROCESS_STOP.
  * Header parsing read "Columns: TIMESTAMP | ..." as a metadata field.
  * Material, nozzle and targets were parsed but never reached the model.

New: event counts, first occurrence, weight and temperature drift, and a compact `digest`
block written for the LLM to read the way an engineer reads a log.
"""
import re
from collections import Counter, OrderedDict

CYCLE_RE = re.compile(
    r"dot_dia=(?P<dia>[\d.]+)um\s+weight=(?P<wt>[\d.]+)mg\s+f_press=(?P<press>[\d.]+)MPa"
    r"\s+temp=(?P<temp>[\d.]+)C")
HEADER_KEYS = {"Equipment ID", "Equipment Model", "Software", "Fluid Delivery", "Nozzle",
               "Material", "Target Dot Size", "Target Dot Weight", "Log Start", "Log End"}


def _pct(a, b):
    return ((b - a) / a) * 100 if a else 0.0


def extract_log_telemetry(log_file_path: str) -> dict:
    with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    metadata = {}
    for line in lines[:40]:
        if ":" in line and "|" not in line and not line.startswith("="):
            k, v = line.split(":", 1)
            if k.strip() in HEADER_KEYS:
                metadata[k.strip()] = v.strip()

    events = []        # (ts, level, event, detail)
    cycles = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
            ts, level, event = parts[0], parts[3].upper(), parts[4]
            detail = " | ".join(parts[5:])
            if level in ("WARN", "ERROR"):
                events.append((ts, level, event, detail))
        m = CYCLE_RE.search(line)
        if m:
            cycles.append({k: float(v) for k, v in m.groupdict().items()})

    counts = Counter((lvl, ev) for _, lvl, ev, _ in events)
    first_seen = OrderedDict()
    for ts, lvl, ev, _ in events:
        first_seen.setdefault((lvl, ev), ts)

    drift_summary = "No cycle data in log."
    drift = {}
    if cycles:
        a, b = cycles[0], cycles[-1]
        drift = {"dia": _pct(a["dia"], b["dia"]), "wt": _pct(a["wt"], b["wt"]),
                 "press": _pct(a["press"], b["press"]), "temp": b["temp"] - a["temp"]}
        drift_summary = (
            f"Dot size {a['dia']}->{b['dia']}um ({drift['dia']:+.1f}%, target "
            f"{metadata.get('Target Dot Size', 'n/a')}) | weight {drift['wt']:+.1f}% | "
            f"fluid pressure {drift['press']:+.1f}% | temp {drift['temp']:+.1f}C")

    recent = [f"{ts[11:19]} {lvl} {ev}" + (f": {d.split(' | ')[0]}" if d else "")
              for ts, lvl, ev, d in events[-5:]]
    has_error = any(lvl == "ERROR" for _, lvl, _, _ in events)

    digest = [
        f"Equipment: {metadata.get('Equipment ID', '?')} ({metadata.get('Equipment Model', '?')}), "
        f"pump {metadata.get('Fluid Delivery', '?')}, nozzle {metadata.get('Nozzle', '?')}, "
        f"material {metadata.get('Material', '?')}",
        f"Log window: {metadata.get('Log Start', '?')} to {metadata.get('Log End', '?')}, "
        f"{len(cycles)} cycle records",
        f"Trend first->last cycle: {drift_summary}",
        "Alarm/warning counts (first seen): " + (", ".join(
            f"{ev} [{lvl}] x{n} (from {first_seen[(lvl, ev)][11:16]})"
            for (lvl, ev), n in counts.most_common(8)) or "none"),
        "Last events: " + ("; ".join(recent) or "none"),
    ]

    return {
        "machine_id": metadata.get("Equipment ID", "Unknown"),
        "model": metadata.get("Equipment Model", "Unknown"),
        "pump": metadata.get("Fluid Delivery", "Unknown"),
        "material": metadata.get("Material", "Unknown"),
        "nozzle": metadata.get("Nozzle", "Unknown"),
        "drift_summary": drift_summary,
        "drift": drift,
        "event_counts": {f"{ev} [{lvl}]": n for (lvl, ev), n in counts.most_common()},
        "recent_alarms": recent or ["No active alarms logged."],
        "digest": "\n".join(digest),
        "status": "CRITICAL" if has_error else "WARNING" if events else "NOMINAL",
    }
