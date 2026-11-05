"""
================================================================================
 NSW DISPENSING MACHINE - FAKE LOG FILE GENERATOR
================================================================================

 WHAT THIS DOES
 --------------
 Real factories keep a log file for every dispensing machine. The machine
 writes one line every time something happens: a dot is dispensed, pressure
 changes, a nozzle gets cleaned, an alarm fires.

 We do not have access to real NSW log files, so this script creates
 realistic FAKE ones. A technician can attach one of these logs to his
 question, and our AI can read it and see what the machine was doing.

 Every number and every event name in here comes from the real NSW
 documents (nFlow software manual, product catalog, pump brochures,
 and the Heraeus / Indium / DELO technical papers).

 HOW TO USE
 ----------
     python generate_machine_logs.py

 It creates a folder called "machine_logs" with one .txt file per machine.

 TO CHANGE THINGS
 ----------------
 Look at the CONFIG section and the MACHINES section below. That is all
 you need to touch.

================================================================================
"""

import os
import random
from datetime import datetime, timedelta

# ==============================================================================
# CONFIG - change these if you want
# ==============================================================================

OUTPUT_FOLDER = "machine_logs"   # where the .txt files are saved
SHIFT_START = datetime(2026, 9, 15, 6, 0, 0)   # when the shift begins
SHIFT_HOURS = 8                  # how long the machine runs
LOG_EVERY_SECONDS = 90           # machine writes a status line this often
RANDOM_SEED = 42                 # same seed = same logs every time (good for testing)

random.seed(RANDOM_SEED)


# ==============================================================================
# MACHINES - the fleet. Add or remove machines here.
# ==============================================================================
# Each machine is a dictionary. The fields mean:
#
#   id          : the name in the factory (what a technician would say)
#   model       : real NSW machine model
#   pump        : real NSW pump fitted to it
#   nozzle      : real NSW nozzle part label
#   nozzle_id_um: nozzle inner diameter in microns
#   material    : what fluid it is dispensing
#   target_um   : the dot size we WANT, in microns
#   target_mg   : the dot weight we WANT, in milligrams
#   f_press_mpa : fluid pressure setting (from nFlow "F Pressure")
#   temp_c      : material stabilizer temperature setting
#   story       : what goes wrong (or not) during the shift - see STORIES below
# ==============================================================================

MACHINES = [
    {
        "id": "NSW-i10-01", "model": "i10 Large Format Micro-Dispenser",
        "pump": "SynchroPULSE-MICRO", "nozzle": "S50-10C", "nozzle_id_um": 100,
        "material": "Solder Paste T6 SAC305 (WS5112)", "target_um": 100,
        "target_mg": 0.0242, "f_press_mpa": 0.70, "temp_c": 24.0,
        "story": "healthy",
    },
    {
        "id": "NSW-N4-02", "model": "N4 Solder Paste Micro-Dispenser",
        "pump": "Time Pressure System", "nozzle": "S50-10C", "nozzle_id_um": 100,
        "material": "Solder Paste T6 SAC305 (WS5112)", "target_um": 220,
        "target_mg": 0.0310, "f_press_mpa": 0.65, "temp_c": 24.0,
        "story": "nozzle_clog",
    },
    {
        "id": "NSW-SD1X-03", "model": "SD1-X Automated Intelligent Micro-Dispenser",
        "pump": "SynchroPULSE-MICRO", "nozzle": "S50-08C", "nozzle_id_um": 80,
        "material": "Solder Paste T6 Sn63Pb37", "target_um": 85,
        "target_mg": 0.0195, "f_press_mpa": 0.72, "temp_c": 23.5,
        "story": "temperature_drift",
    },
    {
        "id": "NSW-W12-04", "model": "W12-2020 Wafer Handling Dispenser",
        "pump": "PistonPRO 4+", "nozzle": "P50-08C", "nozzle_id_um": 80,
        "material": "DELO UV Micro-Dam Adhesive", "target_um": 80,
        "target_mg": 0.0088, "f_press_mpa": 0.55, "temp_c": 25.0,
        "story": "height_sensor_drift",
    },
    {
        "id": "NSW-S420-05", "model": "i-DR S420 Compact Smart Dispenser",
        "pump": "HeliMASTER-3", "nozzle": "H20-15C", "nozzle_id_um": 150,
        "material": "Phosphor Silicone (LED)", "target_um": 300,
        "target_mg": 0.1450, "f_press_mpa": 0.40, "temp_c": 25.0,
        "story": "sedimentation",
    },
    {
        "id": "NSW-A302L-06", "model": "i-DR A302L Dual-Head Automated Dispenser",
        "pump": "PistonPRO 4+", "nozzle": "P50-15C", "nozzle_id_um": 150,
        "material": "Capillary Underfill Epoxy", "target_um": 280,
        "target_mg": 0.0920, "f_press_mpa": 0.60, "temp_c": 24.5,
        "story": "air_pressure_unstable",
    },
]


# ==============================================================================
# REAL NSW SYSTEM ALARMS - copied straight from the nFlow manual
# ==============================================================================
# These are NOT invented. They are the fixed status flags and timeout names
# that exist on EVERY NSW machine, no matter the model or the material.
#
# Group 1: axis status flags, from Chapter 5.1 "Motion"
# Group 2: timeout names, from Chapter 6.2.2 "Timeout Para" (all 5000 ms)
#
# They fire rarely and at random, because in real life a limit switch or a
# servo fault can happen any time on any machine.
# ==============================================================================

AXIS_ALARMS = [
    ("DRIVER_ALARM",  "ALARM", "alarm from driver system"),
    ("POS_ERROR",     "ALARM", "axis position error"),
    ("H_LIMIT_POS",   "ALARM", "hardware positive limit reached"),
    ("H_LIMIT_NEG",   "ALARM", "hardware negative limit reached"),
    ("S_LIMIT_POS",   "EXCEP", "software positive limit reached"),
    ("S_LIMIT_NEG",   "EXCEP", "software negative limit reached"),
]

TIMEOUT_ALARMS = [
    ("GantryX_Timeout",   "EXCEP", "TimeOutProcess=5000ms exceeded"),
    ("GantryY_Timeout",   "EXCEP", "TimeOutProcess=5000ms exceeded"),
    ("GantryZ_Timeout",   "EXCEP", "TimeOutProcess=5000ms exceeded"),
    ("Conveyor1_Timeout", "EXCEP", "TimeOutLoadIn=5000ms exceeded"),
    ("Conveyor2_Timeout", "EXCEP", "TimeOutLoadOut=5000ms exceeded"),
]

# How often a random system alarm shows up (0.004 = about 4 times in 1000 lines)
SYSTEM_ALARM_CHANCE = 0.004


# ==============================================================================
# STORIES - what goes wrong during the shift
# ==============================================================================
# "healthy"             -> nothing wrong, just normal small wobble
# "nozzle_clog"         -> paste slowly blocks the nozzle. Pressure climbs,
#                          volume drops, dots go missing, finally ERROR.
# "sedimentation"       -> heavy filler sinks in the syringe. Dots get lighter
#                          and lighter because only thin liquid comes out.
# "temperature_drift"   -> material stabilizer fails. Fluid gets warm and runny,
#                          so dots come out too big.
# "height_sensor_drift" -> the Z height reading slowly goes wrong, so the
#                          dispense gap is wrong and dot shape goes bad.
# "air_pressure_unstable" -> factory air supply keeps dipping, so a
#                          time/pressure based dot size jumps around.
# ==============================================================================


# ==============================================================================
# HELPER FUNCTIONS - the small tools the generator uses
# ==============================================================================

def wobble(value, percent):
    """Add a small random up/down change, like real sensor noise."""
    return value * (1 + random.uniform(-percent, percent))


def stamp(t):
    """Format the time the way machine logs usually do."""
    return t.strftime("%Y-%m-%d %H:%M:%S")


def line(t, machine_id, level, event, detail="", category="EVENT"):
    """
    Build one log line. This is the shape of every line in the file.

    'category' is the nFlow log tab the line belongs to. The real software
    screen has these tabs: PARA, ALARM, NOTIFY, EVENT, CTRL, EXCEP,
    PROCESS, SYSTEM. We use the ones that make sense for a running job.
    """
    base = (f"{stamp(t)} | {machine_id:<12} | {category:<7} | "
            f"{level:<5} | {event:<22}")
    return base + (f" | {detail}" if detail else "")


def header(m, start, end):
    """The block of text at the top of every log file."""
    return f"""================================================================================
NSW AUTOMATION - DISPENSING EQUIPMENT EVENT LOG
================================================================================
Equipment ID      : {m['id']}
Equipment Model   : {m['model']}
Software          : nFlow Dispensing Software (Windows 10)
Fluid Delivery    : {m['pump']}
Nozzle            : {m['nozzle']}  (inner diameter {m['nozzle_id_um']} um)
Material          : {m['material']}
Target Dot Size   : {m['target_um']} um
Target Dot Weight : {m['target_mg']:.4f} mg
Log Start         : {stamp(start)}
Log End           : {stamp(end)}
================================================================================
Columns: TIMESTAMP | EQUIPMENT | CATEGORY | LEVEL | EVENT | DATA
Category: nFlow log tabs - SYSTEM, PROCESS, EVENT, ALARM, EXCEP, NOTIFY, CTRL
Levels  : INFO = normal, WARN = out of tolerance, ERROR = process stopped
================================================================================

"""


# ==============================================================================
# STARTUP SEQUENCE - what the machine does before it dispenses anything
# ==============================================================================
# This order is taken from the nFlow manual: the software initialises, asks
# for homing, loads the recipe, calibrates the needle, then purges and cleans
# the needle before any dispense command is allowed to run.
# ==============================================================================

def startup_lines(m, t):
    out = []
    # Each step is (event, category, detail). The category is the nFlow tab.
    steps = [
        ("SYSTEM_START_UP", "SYSTEM", f"nFlow software started, recipe=PROD_{m['id'][-2:]}.xml"),
        ("MOTION_HOMING", "SYSTEM", "axis=XYZ status=PASS"),
        ("PUMP_CONNECT", "CTRL", f"pump={m['pump']} status=CONNECTED"),
        ("REGULATOR_SET", "PARA", f"Reg0={m['f_press_mpa']:.2f}MPa Reg1=0.10MPa"),
        ("TEMPERATURE_SET", "PARA", f"stabilizer_setpoint={m['temp_c']:.1f}C"),
        ("NEEDLE_XY_CAL", "EVENT", "offset_x=0.012mm offset_y=-0.008mm status=PASS"),
        ("Z_TOUCH_CAL", "EVENT", "z_touch=12.431mm encoder_res=0.002mm status=PASS"),
        ("DYNAMIC_JET_CAL", "EVENT", "pitch=1.000mm dot_count=1 status=PASS"),
        ("PURGE_STAGE", "EVENT", "purge_count=3 unit_interval=0 status=PASS"),
        ("NEEDLE_CLEAN", "EVENT", "vac_time=800ms post_vac=500ms count=3 status=PASS"),
        ("PAT_ALIGN_BOARD", "EVENT", "pattern_id=0 score=94% theta=0.3deg status=PASS"),
        ("HEIGHT_ALIGN_BOARD", "EVENT", "mode=AVERAGE points=4 offset=0.03mm status=PASS"),
        ("PROCESS_RUN", "PROCESS", "mode=NORMAL run_select=CONTINUOUS"),
    ]
    for event, category, detail in steps:
        out.append(line(t, m["id"], "INFO", event, detail, category))
        t += timedelta(seconds=random.randint(8, 40))
    return out, t


# ==============================================================================
# RANDOM SYSTEM ALARM - the fixed NSW alarms that can hit any machine
# ==============================================================================

def maybe_system_alarm(m, t):
    """
    Now and then, throw one of the real NSW alarms into the log.
    Returns a list: either empty, or one alarm line.
    """
    if random.random() >= SYSTEM_ALARM_CHANCE:
        return []

    if random.random() < 0.6:
        # Axis flag - it does not say which axis, so we add one
        event, category, msg = random.choice(AXIS_ALARMS)
        detail = f"{msg} | axis={random.choice(['X', 'Y', 'Z'])}"
    else:
        # Timeout - the name already says the axis, so do not add one
        event, category, msg = random.choice(TIMEOUT_ALARMS)
        detail = msg

    return [line(t, m["id"], "WARN", event, detail, category)]


# ==============================================================================
# THE STORY ENGINE - decides how the numbers drift as the shift goes on
# ==============================================================================
# "progress" is how far we are through the shift: 0.0 at the start,
# 1.0 at the end. Each story bends the readings in its own way.
#
# It returns 4 things:
#   pressure_factor : multiply the pressure by this
#   volume_factor   : multiply the dot weight by this
#   size_factor     : multiply the dot diameter by this
#   temp_offset     : add this to the temperature, in degrees C
# ==============================================================================

def story_effect(story, progress):
    if story == "healthy":
        return 1.0, 1.0, 1.0, 0.0

    if story == "nozzle_clog":
        # Paste packs into the nozzle. Machine pushes harder, gets less out.
        if progress < 0.45:
            return 1.0, 1.0, 1.0, 0.0
        bad = (progress - 0.45) / 0.55          # 0 -> 1 over the second half
        return 1 + 0.75 * bad, 1 - 0.55 * bad, 1 - 0.30 * bad, 0.0

    if story == "sedimentation":
        # Heavy filler sinks. Thin liquid on top means lighter dots.
        return 1.0, 1 - 0.28 * progress, 1 - 0.10 * progress, 0.0

    if story == "temperature_drift":
        # Stabilizer fails. Warm fluid is runny, so more comes out.
        if progress < 0.3:
            return 1.0, 1.0, 1.0, 0.0
        bad = (progress - 0.3) / 0.7
        return 1.0, 1 + 0.35 * bad, 1 + 0.22 * bad, 6.5 * bad

    if story == "height_sensor_drift":
        # Wrong dispense gap. Dot spreads wider and gets messy.
        bad = progress
        return 1.0, 1 + 0.05 * bad, 1 + 0.26 * bad, 0.0

    if story == "air_pressure_unstable":
        # Supply keeps dipping. Nothing trends, it just jumps around.
        jump = random.choice([0.86, 0.94, 1.0, 1.0, 1.08, 1.16])
        return jump, jump, (1 + (jump - 1) * 0.6), 0.0

    return 1.0, 1.0, 1.0, 0.0


# ==============================================================================
# ALARM MESSAGES - the words the machine writes when the story goes wrong
# ==============================================================================

def story_alarm(story, severity):
    """severity is 'warn' or 'error'."""
    messages = {
        "nozzle_clog": {
            "warn":  ("DISPENSE_VOLUME_LOW", "dispensed volume below lower spec limit"),
            "error": ("NOZZLE_CLOG_DETECTED", "no material detected at nozzle, dispensing halted"),
        },
        "sedimentation": {
            "warn":  ("WEIGHT_TREND_DOWN", "weigh scale trending below target, check agitation"),
            "error": ("MATERIAL_INHOMOGENEOUS", "filler separation suspected in syringe"),
        },
        "temperature_drift": {
            "warn":  ("TEMP_OUT_OF_RANGE", "material stabilizer above setpoint tolerance"),
            "error": ("STABILIZER_FAULT", "temperature control loop not responding"),
        },
        "height_sensor_drift": {
            "warn":  ("HEIGHT_OFFSET_HIGH", "height sensor reading outside accepted range"),
            "error": ("DISPENSE_GAP_FAIL", "dispense gap out of tolerance, recalibration required"),
        },
        "air_pressure_unstable": {
            "warn":  ("AIR_PRESSURE_DIP", "incoming air supply below 5 Bar"),
            "error": ("REGULATOR_UNSTABLE", "fluid pressure cannot hold setpoint"),
        },
    }
    return messages.get(story, {}).get(severity, ("GENERIC_ALARM", "unspecified condition"))


# ==============================================================================
# THE MAIN GENERATOR - builds the whole log for one machine
# ==============================================================================

def build_log(m):
    t = SHIFT_START
    end_time = SHIFT_START + timedelta(hours=SHIFT_HOURS)
    lines = []

    # ---- 1. Machine boots up and calibrates -------------------------------
    boot_lines, t = startup_lines(m, t)
    lines.extend(boot_lines)

    # ---- 2. Set up the running counters -----------------------------------
    cycle = 0                 # how many units dispensed so far
    board = 1                 # which board we are on
    warned = False            # have we printed the WARN line yet?
    errored = False           # have we printed the ERROR line yet?
    total_seconds = SHIFT_HOURS * 3600

    # Tolerance: how far from target is still acceptable (+/- 20 um, as in
    # the Indium paper's Cpk study)
    size_low = m["target_um"] - 20
    size_high = m["target_um"] + 20

    # ---- 3. The main loop: one pass = one status line ---------------------
    while t < end_time and not errored:
        progress = (t - SHIFT_START).total_seconds() / total_seconds
        p_fac, v_fac, s_fac, t_off = story_effect(m["story"], progress)

        # Work out this moment's readings
        pressure = wobble(m["f_press_mpa"] * p_fac, 0.02)
        weight = wobble(m["target_mg"] * v_fac, 0.04)
        size = wobble(m["target_um"] * s_fac, 0.03)
        temp = wobble(m["temp_c"] + t_off, 0.005)
        cycle += random.randint(40, 70)

        data = (f"cycle={cycle} dot_dia={size:.1f}um weight={weight:.4f}mg "
                f"f_press={pressure:.3f}MPa temp={temp:.1f}C board={board}")

        # Decide if this reading is good, warning, or a stop
        if size_low <= size <= size_high:
            lines.append(line(t, m["id"], "INFO", "DISPENSE_CYCLE_OK",
                              data, "PROCESS"))
        else:
            # Out of tolerance. First time, print the story's warning.
            if not warned:
                event, msg = story_alarm(m["story"], "warn")
                lines.append(line(t, m["id"], "WARN", event,
                                  f"{msg} | {data}", "NOTIFY"))
                warned = True
            else:
                lines.append(line(t, m["id"], "WARN", "DOT_SIZE_OUT_OF_SPEC",
                                  f"spec={size_low}-{size_high}um | {data}",
                                  "NOTIFY"))

            # If it drifts really far, the machine stops.
            if size < size_low * 0.85 or size > size_high * 1.15:
                event, msg = story_alarm(m["story"], "error")
                lines.append(line(t, m["id"], "ERROR", event,
                                  f"{msg} | {data}", "ALARM"))
                lines.append(line(t, m["id"], "ERROR", "PROCESS_STOP",
                                  f"operator intervention required, cycle={cycle}",
                                  "PROCESS"))
                errored = True

        # ---- Real NSW system alarms - can hit any machine, any time -------
        lines.extend(maybe_system_alarm(m, t))

        # ---- Routine housekeeping the machine does by itself --------------
        if cycle % 500 < 60:
            lines.append(line(t, m["id"], "INFO", "NEEDLE_CLEAN",
                              "vac_time=800ms post_vac=500ms count=3 status=PASS",
                              "EVENT"))
        if cycle % 1200 < 60:
            lines.append(line(t, m["id"], "INFO", "PURGE_STAGE",
                              "purge_count=3 status=PASS", "EVENT"))
        if cycle % 900 < 60:
            board += 1
            lines.append(line(t, m["id"], "INFO", "BOARD_COMPLETE",
                              f"board={board - 1} units=900 defects=0", "PROCESS"))
            # Vision align sometimes fails a board and the unit gets skipped
            # (nFlow "Skip Prompt" bypasses a FAIL unit - Chapter 7.4.1)
            score = random.randint(70, 97)
            if score < 78:
                lines.append(line(t, m["id"], "WARN", "PAT_ALIGN_BOARD",
                                  f"pattern_id=0 score={score}% status=FAIL "
                                  f"skip_prompt=TRUE unit_bypassed", "EXCEP"))
            else:
                lines.append(line(t, m["id"], "INFO", "PAT_ALIGN_BOARD",
                                  f"pattern_id=0 score={score}% status=PASS",
                                  "EVENT"))

        # Height sensor out-of-range prompt (nFlow HSensor config, Ch 6.1.4)
        if m["story"] == "height_sensor_drift" and progress > 0.5 \
                and random.random() < 0.05:
            lines.append(line(t, m["id"], "WARN", "HSENSOR_OUT_OF_RANGE",
                              "poll reading outside Out-of-Range Value", "NOTIFY"))

        # Occasional missing dot - happens more once things go bad
        miss_chance = 0.02 if not warned else 0.18
        if random.random() < miss_chance:
            lines.append(line(t + timedelta(seconds=5), m["id"], "WARN",
                              "MISSING_DOT",
                              f"unit skipped, cycle={cycle} vision_score=LOW",
                              "NOTIFY"))

        t += timedelta(seconds=LOG_EVERY_SECONDS + random.randint(-15, 15))

    # ---- 4. How the shift finished ----------------------------------------
    if not errored:
        lines.append(line(t, m["id"], "INFO", "PROCESS_END",
                          f"shift complete, total_cycles={cycle}, boards={board}",
                          "PROCESS"))
    lines.append(line(t + timedelta(seconds=30), m["id"], "INFO", "SYSTEM_SHUTDOWN",
                      "operator logged out", "SYSTEM"))

    return header(m, SHIFT_START, t), lines


# ==============================================================================
# RUN IT
# ==============================================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Writing logs into {os.path.abspath(OUTPUT_FOLDER)}\n")

    for m in MACHINES:
        head, lines = build_log(m)
        path = os.path.join(OUTPUT_FOLDER, f"{m['id']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(head)
            f.write("\n".join(lines))
            f.write("\n")

        warns = sum(1 for l in lines if "| WARN " in l)
        errors = sum(1 for l in lines if "| ERROR" in l)
        print(f"  {m['id']:<14} story={m['story']:<22} "
              f"lines={len(lines):<5} warnings={warns:<4} errors={errors}")

    print(f"\nDone. {len(MACHINES)} log files created.")
    print(f"Saved to: {os.path.abspath(OUTPUT_FOLDER)}")


if __name__ == "__main__":
    main()
