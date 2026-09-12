# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file generalize_speech_signals.py
# @brief GENERALIZE the flagship signals-vs-speech ablation to every verbatim air-to-ground fault report (7 atlas exchange_quotes + 14 cross-mission found_exch.
#
# ange). For each: extract the CREW's report (drop CapCom lines so the
# ground's diagnosis isn't fed back), split into SIGNALS (CAPS annunciators +
# readings + documented annunciator vocab) vs SPEECH (plain-language
# remainder), run diagnose under FULL/SIGNALS/SPEECH, score subsystem-match.
# Reports subsystem accuracy per condition. DEMO=1 prints the crew-text +
# split (no LLM). Run from analysis/; CUDA_VISIBLE_DEVICES=0.
#
"""GENERALIZE the flagship signals-vs-speech ablation to every verbatim air-to-ground fault report
(7 atlas exchange_quotes + 14 cross-mission found_exchange). For each: extract the CREW's report
(drop CapCom lines so the ground's diagnosis isn't fed back), split into SIGNALS (CAPS annunciators +
readings + documented annunciator vocab) vs SPEECH (plain-language remainder), run diagnose under
FULL/SIGNALS/SPEECH, score subsystem-match. Reports subsystem accuracy per condition.
DEMO=1 prints the crew-text + split (no LLM). Run from analysis/; CUDA_VISIBLE_DEVICES=0."""
import sys
import json, os, re, sys
sys.path.insert(0, "scripts")
from subsystem_match import aliases_for, names, subsystem_match
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge import retriever as R
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever, tokenize
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain
TE = os.environ.get("LODESTAR_ATLAS_DIR", "../../apollo-anomaly-atlas/data")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434"); MODEL = os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b")

# ---- collect cases: (subsystem, exchange_quote) ----
atlas = json.load(open(f"{TE}/atlas_parsed.json", encoding="utf-8"))
cases = [("atlas", a.get("subsystem"), a["exchange_quote"])
         for m in atlas["missions"] for a in m.get("anomalies", []) if str(a.get("exchange_quote") or "").strip()]
# cross-mission subsystem via the run_capcom_crossmission ASSIGN map
sys.path.insert(0, "scripts")
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402

spec = importlib.util.spec_from_file_location("rcc", "scripts/run_capcom_crossmission.py")
# avoid importing (it runs Ollama on import); instead inline the ASSIGN map:
ASSIGN = [("FDAI attitude jumped","GNC"),("docking probe would not extend","structures"),
 ("cabin pressure dropped rapidly after LM jettison","structures"),("1202 and 1201 program alarms","GNC"),
 ("lightning strike at launch","EPS"),("docking probe will not capture","structures"),
 ("landing radar will not lock","GNC"),("SPS thrust light on the EMS","SPS"),
 ("water leaking from the chlorine","ECLSS"),("AC bus 2 and DC bus B undervolt","EPS"),
 ("SPS secondary yaw gimbal servo","SPS"),("erroneous gimbal-lock indication","GNC"),
 ("spurious master alarms with no caution","EPS"),("EMS accelerometer null-bias shift","GNC")]
def assign(sym):
    for sub, ss in ASSIGN:
        if sub.lower() in (sym or "").lower(): return ss
    return None
cm = json.load(open(f"{TE}/crossmission_human_turns.json", encoding="utf-8"))["anomalies"]
for a in cm:
    if a.get("found_exchange") and str(a.get("exchange_quote") or "").strip():
        cases.append(("crossmission", assign(a["symptom"]), a["exchange_quote"]))

# ---- crew-report extraction: drop CapCom-tagged + instruction (fix) lines + editorial annotations ----
GROUND = re.compile(r"\b(CC|CAPCOM|CAP\s*COM|HOUSTON)\b|\(cap ?com\)", re.I)
# editorial annotation labels embedded in the quotes (not crew speech)
ANNOT = re.compile(r"\b(CREW REPORT|CREW FAULT|CLARIFYING QUESTION|FIRST CORRECTIVE INSTRUCTION|"
    r"CORRECTIVE INSTRUCTION|FTI|abnormal let|ground then|contractor|simulations|rev \d)\b", re.I)
# CapCom fixes/instructions (drop regardless of speaker tag — these leak the diagnosis)
INSTR = re.compile(r"^(we'?d like|we would like|would you|could you|can you|cycle|pull|verify|suggest|"
    r"go ahead|copy|roger|understand|we're go|we got you|okay\.? copy)", re.I)
SEG = re.compile(r"(?:\d{2,3}:\d{2}:\d{2}|\(\d+\)|\||->| / )")
def crew_text(ex):
    ex = re.sub(r"\[[^\]]*\]", " ", ex)                        # drop [bracketed] editorial inserts
    ex = re.sub(r"^.*?crew report[^:]*:", "", ex, flags=re.I)
    keep = []
    for p in SEG.split(ex):
        p = p.strip()
        if not p or len(p) < 5: continue
        m = re.match(r"([A-Za-z()/ .]+?):\s*(.*)", p, re.S)
        spk, txt = (m.group(1), m.group(2)) if m else ("", p)
        if GROUND.search(spk): continue                        # drop ground-tagged lines
        txt = txt.strip().strip("'\"").strip()
        if not txt or INSTR.match(txt): continue               # drop acks / fix instructions
        txt = ANNOT.sub(" ", txt)                              # strip embedded editorial labels
        txt = re.sub(r"\([^)]*\)", " ", txt).strip()          # drop (parenthetical) speaker/prose
        if txt: keep.append(txt)
    return re.sub(r"\s+", " ", " ".join(keep)).strip()

# pre-registered PANEL / annunciator lexicon (fixed; not tuned to the outcome)
PANEL = set("bus buses undervolt undervoltage volt volts voltage amp amps alarm master caution warning "
    "light lights quantity pressure psi temperature temp gimbal program radar servo thrust attitude "
    "platform transponder telemetry valve breaker breakers helium fuel cell cells oxygen o2 ac dc ems "
    "pgncs pgns gdc fdai cdu rcs sps sce imu accelerometer barber pole bit abort circuit inverter "
    "restart reset undervolts overload disconnect".split())
_STOP = set("the a an of to in on and or with at from for was were is are be been being this that these those "
    "it its they them their has had have will would could should may might not no nor due most probable "
    "we i he she our your okay roger copy yes now here there just got get right thing some any all only "
    "looking look see saw like about that's we've i'm it's you sir jack gene ed houston well think need".split())
def split(text):
    caps = [m.group().strip() for m in R._CAPS.finditer(text or "")]
    caps = [c for c in caps if len(c) >= 2 and c.upper() not in R._STOP and not c.isdigit()]
    capw = set(w.lower() for c in caps for w in re.split(r"[^A-Za-z0-9]+", c) if w)
    sig, spe = [], []
    for t in tokenize(text or ""):
        tl = t.lower()
        if tl in _STOP or len(tl) < 2: continue
        if tl in PANEL or tl in capw or any(ch.isdigit() for ch in t): sig.append(t)
        else: spe.append(t)
    return " ".join(caps + [w for w in sig if w.lower() not in capw]).strip(), " ".join(spe).strip()

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
r = SalienceBM25Retriever(segs); g = CorpusGraph(r.segments); ANN = g.ann_vocab

rows = [(src, sub, crew_text(ex)) for src, sub, ex in cases if sub]  # need a ground-truth subsystem
if os.environ.get("DEMO"):
    for src, sub, ct in rows:
        s, p = split(ct)
        print(f"[{src}/{sub}] CREW: {ct[:120]}")
        print(f"     SIG: {s[:100]}")
        print(f"     SPE: {p[:100]}\n")
    print(f"total scorable cases: {len(rows)}"); sys.exit(0)

diagnose_chain("MAIN B BUS UNDERVOLT", r, g, model=MODEL, base_url=BASE, k_entries=8, pool=16)  # warm
tally = {c: {"n": 0, "ok": 0} for c in ("FULL", "SIGNALS", "SPEECH")}
OUT = open(os.environ.get("OUT", _d("generalize_speech_signals.jsonl")), "w", encoding="utf-8")
for i, (src, sub, ct) in enumerate(rows, 1):
    s, p = split(ct); rec = {"src": src, "sub": sub, "crew": ct[:160]}
    for cond, q in (("FULL", ct), ("SIGNALS", s), ("SPEECH", p)):
        if not q.strip(): rec[cond] = False; tally[cond]["n"] += 1; continue
        d = diagnose_chain(q, r, g, model=MODEL, base_url=BASE, k_entries=8, pool=16)
        ok = subsystem_match("" if d["declined"] else d.get("diagnosis", ""), sub, d["declined"])
        rec[cond] = bool(ok); tally[cond]["n"] += 1; tally[cond]["ok"] += bool(ok)
    OUT.write(json.dumps(rec) + "\n"); OUT.flush()
    print(f"[{i}/{len(rows)}] {sub:<11} FULL={rec['FULL']} SIG={rec['SIGNALS']} SPE={rec['SPEECH']}", flush=True)
OUT.close()
print("\n===== GENERALIZED SIGNALS-vs-SPEECH (subsystem match) =====")
for c in tally:
    n = tally[c]["n"] or 1
    print(f"  {c:<8}: {tally[c]['ok']}/{n} = {tally[c]['ok']/n:.0%}")
