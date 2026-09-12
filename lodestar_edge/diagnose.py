

# --- offline enforcement ------------------------------------------------------
# LODESTAR is documented as offline and on-device, so the generation endpoint is
# restricted to loopback unless the operator explicitly opts out. Without this,
# OLLAMA_URL / --base-url accepts any host and crew symptoms plus retrieved
# manual excerpts are POSTed to it.
# Sentinel returned by _chat when every attempt failed. Callers MUST treat a result
# carrying this as UNAVAILABLE (the model could not be reached), which is a different
# state from DECLINED (the manuals do not cover the symptom).
UNAVAILABLE = "(generation unavailable:"

_LOOPBACK = ("http://localhost", "http://127.0.0.1", "http://[::1]",
             "https://localhost", "https://127.0.0.1", "https://[::1]")


def assert_local_endpoint(base_url: str) -> str:
    """@brief Refuse a non-loopback generation endpoint unless explicitly allowed.

    @param base_url The Ollama base URL about to be used.
    @return @p base_url unchanged, if permitted.
    @exception ValueError if the endpoint is remote and LODESTAR_ALLOW_REMOTE != "1".
    """
    if os.environ.get("LODESTAR_ALLOW_REMOTE") == "1":
        return base_url
    if not str(base_url).startswith(_LOOPBACK):
        raise ValueError(
            "refusing a non-loopback generation endpoint: %r. LODESTAR is offline "
            "and on-device; crew symptoms and manual excerpts must not leave it. "
            "Set LODESTAR_ALLOW_REMOTE=1 to override deliberately." % (base_url,))
    return base_url

# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file diagnose.py
# @brief Cite-LEAD-first diagnosis, point-by-point over the retrieved sources, with two DETERMINISTIC, statically-auditable dials (no extra model calls; every .
#
# input already exists when `points` is built): * CITABILITY (per lead) —
# `extr` = fraction of the lead's content terms that actually appear in the
# source it cites. Below EXTR_WEAK the lead is flagged "weakly sourced —
# verify against source". * CITE-SCORE (per diagnosis) — a coverage-suspicion
# dial: z(groundedness g) + z(manual-support sup), where g = mean extr and sup
# = how many retrieved sources' text contain the top agreement term. Low CITE-
# SCORE => the leads may lie outside the onboard manuals. TRIAGE, NOT A
# DETECTOR: at any useful cut ~half of flags are false (independently
# validated in/out AUC ~0.69, permutation p~0.0015, on the k=8 full_log run —
# see analysis/cite_score_validate.py). Never present it as a coverage
# guarantee. * AGREEMENT (per diagnosis) — cross-source consensus, ECHO-
# EXCLUDED (symptom-salience terms removed so the vote reflects a named
# component, not shared symptom vocabulary). Reliability dial only. Design
# intent (a search-offloader with a HUMAN VERIFIER should surface leads, not
# go silent): * Each source is judged in its OWN tiny prompt (symptom + one
# source): a candidate cause, or SKIP. * LEAD-FIRST — present a candidate from
# every source the model finds relevant, cited to that source. Uncited
# candidates are kept and marked "inferred", not dropped. Out-of-corpus faults
# are surfaced too, flagged by CITE-SCORE. DECLINE only when there is no fault
# indication / no source matched. * The numeric 0-1 "confidence" was removed:
# a decimal reads as a probability and nothing measured supports that. The
# crew sees an agreement line + per-lead markers + a coverage warning instead.
#
"""Cite-LEAD-first diagnosis, point-by-point over the retrieved sources, with two DETERMINISTIC,
statically-auditable dials (no extra model calls; every input already exists when `points` is built):

  * CITABILITY (per lead) — `extr` = fraction of the lead's content terms that actually appear in
    the source it cites. Below EXTR_WEAK the lead is flagged "weakly sourced — verify against source".
  * CITE-SCORE (per diagnosis) — a coverage-suspicion dial: z(groundedness g) + z(manual-support sup),
    where g = mean extr and sup = how many retrieved sources' text contain the top agreement term.
    Low CITE-SCORE => the leads may lie outside the onboard manuals. TRIAGE, NOT A DETECTOR: at any
    useful cut ~half of flags are false (independently validated in/out AUC ~0.69, permutation
    p~0.0015, on the k=8 full_log run — see analysis/cite_score_validate.py). Never present
    it as a coverage guarantee.
  * AGREEMENT (per diagnosis) — cross-source consensus, ECHO-EXCLUDED (symptom-salience terms removed
    so the vote reflects a named component, not shared symptom vocabulary). Reliability dial only.

Design intent (a search-offloader with a HUMAN VERIFIER should surface leads, not go silent):
  * Each source is judged in its OWN tiny prompt (symptom + one source): a candidate cause, or SKIP.
  * LEAD-FIRST — present a candidate from every source the model finds relevant, cited to that source.
    Uncited candidates are kept and marked "inferred", not dropped. Out-of-corpus faults are surfaced
    too, flagged by CITE-SCORE. DECLINE only when there is no fault indication / no source matched.
  * The numeric 0-1 "confidence" was removed: a decimal reads as a probability and nothing measured
    supports that. The crew sees an agreement line + per-lead markers + a coverage warning instead.
"""
import os
import re
import time
from collections import Counter
import requests
from .scope_gate import out_of_domain
from .retriever import salience

# --- robust Ollama transport ------------------------------------------------------------------------
# Bounded context so the KV-cache allocation is predictable. Default 8192 for the dev box; OVERRIDE to
# a smaller value on the 8 GB Jetson (LODESTAR_NUM_CTX=4096) — a large num_ctx is multiplied by Ollama's
# parallel slots and, on the 8 GB board with zram-only swap, drove a memory-thrash livelock that the
# hardware watchdog reset (see the paper's results section). Pair with OLLAMA_NUM_PARALLEL=1 on the Jetson.
NUM_CTX = int(os.environ.get("LODESTAR_NUM_CTX", "8192"))
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def preload(base_url: str, model: str = "llama3.2:3b") -> bool:
    """Force the model resident with a 1-token request. Cheap; call before a batch and to recover
    after an eviction. Returns True if the server answered."""
    try:
        r = _session().post(f"{base_url.rstrip('/')}/api/chat", json={
            "model": model, "stream": False, "keep_alive": -1,
            "messages": [{"role": "user", "content": "ok"}],
            "options": {"num_predict": 1, "num_ctx": 256}}, timeout=(10, 240))
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def unload(base_url: str, model: str = "llama3.2:3b") -> None:
    """Force Ollama to release the model AND its KV cache (keep_alive=0 unloads immediately). Used to
    reset the known Ollama KV-accumulation (ollama/ollama#10114, #16698) on the 8 GB Jetson between
    batches; the next preload() cold-loads it back with a fresh cache."""
    try:
        _session().post(f"{base_url.rstrip('/')}/api/chat", json={
            "model": model, "stream": False, "keep_alive": 0,
            "messages": [{"role": "user", "content": "ok"}],
            "options": {"num_predict": 1, "num_ctx": 256}}, timeout=(10, 120))
    except requests.exceptions.RequestException:
        pass


def _chat(base_url: str, model: str, system: str, user: str, num_predict: int,
          retries: int = 4) -> str:
    """One chat completion, hardened: session reuse, retry-with-backoff on any transport/JSON error
    OR an empty body (a transient Ollama blip returns 200 + empty content), and a model re-preload
    between attempts in case it was evicted under memory pressure. Returns the text, or a sentinel
    '(generation unavailable: ...)' string only after every attempt fails — never raises."""
    url = f"{assert_local_endpoint(base_url).rstrip('/')}/api/chat"
    payload = {
        "model": model, "stream": False, "keep_alive": -1,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict, "num_ctx": NUM_CTX},
    }
    last = "unknown"
    for attempt in range(retries):
        try:
            r = _session().post(url, json=payload, timeout=(10, 240))
            r.raise_for_status()
            txt = ((r.json().get("message") or {}).get("content") or "").strip()
            if txt:
                return txt
            last = "empty"
        except (requests.exceptions.RequestException, ValueError) as e:
            last = type(e).__name__
        if attempt < retries - 1:
            preload(base_url, model)          # model may have been evicted — bring it back
            time.sleep(2 * (attempt + 1))     # linear backoff: 2s, 4s, 6s
    return f"{UNAVAILABLE} {last})"


DECLINE = "No fault indication to work from / no onboard source matched; defer to ground / manual procedure."
DECLINE_DOMAIN = "This is outside the systems my onboard manuals cover ({}); defer to ground / manual procedure."

# --- CITE-SCORE calibration -------------------------------------------------------------------------
# FROZEN from the 173 answered diagnoses of the k=8 full_log.jsonl run (2026-07-11). Reproduced
# independently (substring `sup`, 4+char alpha terms) in analysis/cite_score_validate.py.
# WARNING: these are k=8-calibrated. `sup` grows with k, so the deployed default k=6 shifts the
# distribution — re-fit MU_SUP/SD_SUP and CITE_FLAG from a k=6 re-run before quoting operating-point
# numbers in the paper (IMPLEMENTATION_NOTE gotcha #1 / acceptance check #3).
_MU_G, _SD_G, _MU_SUP, _SD_SUP = 0.3745, 0.1408, 1.6012, 2.0534
CITE_FLAG = -1.510        # ~20th pct: coverage-suspect cut. Triage dial — ~half of flags are false.
EXTR_WEAK = 0.40          # per-lead: below this the lead is "weakly sourced — verify against source".
CONSENSUS_CUT = 3         # k=6: >= this many sources converging => show the agreement line.

POINTWISE_SYSTEM = (
    "You are Lodestar, an onboard engineering-cognition assistant, strictly advisory — you never "
    "act; the crew acts. You are shown a crew symptom and ONE manual source segment. If — and "
    "ONLY if — the source is relevant to the symptom, output ONE concrete candidate cause or "
    "next check drawn from that source, and END your line with the citation tag you are given. "
    "If the source is not relevant to the symptom, output exactly: SKIP"
)

# generic prose / engineering filler that carries no discriminative component info (excluded from
# the agreement vote and from extractiveness, so the signals reflect the actual named component,
# not shared boilerplate or junk-convergence terms).
_GEN = set("the a an of to in on and or with at from for during after before was were is are be this "
    "that these those it its they them their has had have will would could should may might not no non "
    "due most probable caused cause failure failed fault anomaly report system module assembly unit "
    "device component part condition normal nominal operation over under within candidate check next "
    "based given symptom source potential possible verify may causing related issue reading high low "
    "value indicates suggests one concrete drawn ending output relevant crew "
    "which whose indicated potentially malfunction insufficient".split())   # junk-convergence terms
_TERM = re.compile(r"[a-z][a-z\-]{3,}")
_TAG = re.compile(r"\[S\d+\]", re.I)


def _content_terms(t):
    return {w for w in _TERM.findall((t or "").lower()) if w not in _GEN}


def _dials(points, symptom, scored):
    """Compute per-lead extractiveness (mutates points) + diagnosis-level CITE-SCORE and echo-excluded
    consensus. No model calls. NOT fully deterministic: ties in the consensus vote break on set
    iteration order, so PYTHONHASHSEED can change `consensus_term`, `support`, `cite_score` and the
    crew-visible coverage flag. Set PYTHONHASHSEED=0 for a repeatable run. Returns a dict of the fields
    to surface. `scored` = [(seg,score)]."""
    echo = _content_terms(salience(symptom))
    vote = Counter()
    extrs = []
    for p in points:
        o = _content_terms(_TAG.sub("", p["text"]))
        src = _content_terms(p["src_text"])
        extr = len(o & src) / len(o) if o else 0.0
        p["extr"] = round(extr, 3)
        p["weakly_sourced"] = extr < EXTR_WEAK
        extrs.append(extr)
        for w in (o - echo):
            vote[w] += 1
    g = sum(extrs) / len(extrs) if extrs else 0.0
    top, consensus = (vote.most_common(1)[0] if vote else (None, 0))
    # sup = how many RETRIEVED sources' raw text contain the top agreement term (substring)
    sup = sum(1 for (seg, _) in scored if top and top in (seg.raw_text or "").lower()) if top else 0
    cite_score = (g - _MU_G) / _SD_G + (sup - _MU_SUP) / _SD_SUP
    coverage_flag = cite_score <= CITE_FLAG
    n = len(points)
    if consensus >= CONSENSUS_CUT and top:
        agreement = f"agreement: {consensus} of {n} sources converge on '{top}'"
    else:
        agreement = ("low agreement — leads scatter; treat as uncertain, verify each"
                     + (f" (top term '{top}', {consensus}/{n})" if top else ""))
    return {"cite_score": round(cite_score, 3), "coverage_flag": coverage_flag,
            "groundedness": round(g, 3), "support": sup,
            "consensus": consensus, "consensus_term": top, "agreement": agreement}


def _declined(diagnosis, gate=False):
    """@brief Uniform declined result for the pointwise path.
    @param diagnosis Human-facing explanation of why nothing is offered.
    @param gate      True if the scope gate caused the decline.
    @return dict with declined=True; `unavailable` distinguishes a backend outage.
    """
    return {"declined": True, "unavailable": False, "diagnosis": diagnosis,
            "sources": [], "n_points": 0,
            "n_grounded": 0, "n_inferred": 0, "n_dropped_uncited": 0,
            "cite_score": None, "coverage_flag": False, "groundedness": None, "support": 0,
            "consensus": 0, "consensus_term": None, "agreement": None, "gate_declined": gate}


# =====================================================================================================
# GRAPH-CHAIN diagnosis (current design): BM25 finds entry manuals; we CHASE the links_to graph to
# build one reasoning chain-of-causes per manual; the LLM decides over the chains + manuals in ONE call.
# =====================================================================================================
COVERAGE_CUT = 0.38    # symptom-annunciation coverage below this => likely outside onboard manuals

CHAIN_SYSTEM = (
    "You are Lodestar, an onboard engineering-cognition assistant, strictly advisory — you never act; "
    "the crew acts. You are given a crew SYMPTOM and several REASONING CHAINS extracted from the onboard "
    "manuals. Each chain starts at a manual section whose annunciation matches the symptom, then follows "
    "the manuals' OWN documented connections (this indication -> this system -> connected components). "
    "Using ONLY the chains provided, decide the most likely candidate cause(s) to CHECK, and justify each "
    "by walking the chain that leads to it, citing steps as [C<chain>.<step>]. Do not invent components "
    "that are not in the chains. If the chains do not support a cause, say so and defer to ground / "
    "manual procedure. You narrow the search; the crew keeps judgment and acts."
)


def _render_chains(chains):
    """chains: list of lists of Segment. Return (prompt_block, structured) with [C i.j] tags."""
    lines, structured = [], []
    for i, chain in enumerate(chains, 1):
        entry = chain[0]
        lines.append(f"\nCHAIN {i} — entry manual: {entry.source_doc}")
        steps = []
        for j, seg in enumerate(chain, 1):
            tag = f"[C{i}.{j}]"
            label = (seg.component or seg.signature or "").strip()
            label = " ".join(label.split())[:90] or "(section)"
            snip = " ".join((seg.raw_text or "").split())[:180]
            arrow = "" if j == 1 else "  -> "
            lines.append(f"  {arrow}{tag} {label} :: {snip}")
            steps.append({"tag": tag, "label": label, "doc": seg.source_doc,
                          "snippet": snip[:120], "id": seg.id})
        structured.append({"entry_doc": entry.source_doc, "n_steps": len(chain), "steps": steps})
    return "\n".join(lines), structured


def diagnose_chain(symptom: str, retriever, graph, model: str = "llama3.2:3b",
                   base_url: str = "http://localhost:11434", k_entries: int = 8, pool: int = 16,
                   rel_threshold: float = 0.4, max_depth: int = 3, gate: bool = False,
                   chase_criterion: str = "bm25") -> dict:
    """Current pipeline: salience+BM25 entry manuals -> one graph chain per manual -> single LLM
    decision over the chains. chase_criterion: 'bm25' (max symptom-relevance per hop, validated) or
    'coverage' (submodular, max marginal new coverage — under eval). Coverage dial = annunciation match."""
    if gate:
        ood = out_of_domain(symptom)
        if ood:
            return _chain_declined(DECLINE_DOMAIN.format(ood), gate=True)
    ranked = [seg for seg, _ in retriever.search_scored(symptom, k=pool)]
    if not ranked:
        return _chain_declined(DECLINE)
    entries = graph.pick_entries(ranked, k_entries)
    scores = retriever.effective_scores(symptom)   # salience, or the raw-symptom fallback
    if chase_criterion == "coverage":
        chains = [graph.chase_coverage(e, scores, retriever, max_depth=max_depth) for e in entries]
    elif chase_criterion == "coverage_guarded":
        chains = [graph.chase_coverage(e, scores, retriever, rel_floor=0.2, max_depth=max_depth)
                  for e in entries]
    else:
        chains = [graph.chase(e, scores, retriever, rel_threshold=rel_threshold, max_depth=max_depth)
                  for e in entries]
    top1 = retriever.score_of(scores, entries[0])
    coverage = graph.coverage(symptom)
    prompt_block, structured = _render_chains(chains)
    user = (f"SYMPTOM: {symptom}\n{prompt_block}\n\n"
            f"Give: (1) the most likely candidate cause(s) to CHECK, each with the chain steps that "
            f"support it; (2) if uncertain, which chain is most promising and what to verify next.")
    decision = _chat(base_url, model, CHAIN_SYSTEM, user, num_predict=400)
    if decision.startswith(UNAVAILABLE):
        # The model was unreachable. This is NOT a diagnosis and NOT a coverage
        # decline — report it as its own state so a caller (and the crew) can tell
        # "defer to ground because the manuals do not cover this" from "the onboard
        # model is down".
        out = _chain_declined(decision, gate=False)
        out["unavailable"] = True
        return out
    return {"declined": False, "unavailable": False, "diagnosis": decision, "chains": structured,
            "coverage": round(coverage, 3), "coverage_flag": coverage < COVERAGE_CUT,
            "retrieval_strength": round(top1, 1), "n_chains": len(chains),
            "chain_lengths": [c["n_steps"] for c in structured]}


def _chain_declined(diagnosis, gate=False):
    """@brief Uniform declined result. Schema matches the answered result's keys.
    @param diagnosis Human-facing explanation of why nothing is offered.
    @param gate      True if the scope gate caused the decline.
    @return dict with declined=True and every key an answered result carries.
    """
    return {"declined": True, "unavailable": False, "diagnosis": diagnosis,
            "chains": [], "coverage": None, "coverage_flag": False,
            "retrieval_strength": 0.0, "n_chains": 0,
            "chain_lengths": [], "gate_declined": gate}


def diagnose(symptom: str, retriever, model: str = "llama3.2:3b",
             base_url: str = "http://localhost:11434", k: int = 6, gate: bool = False) -> dict:
    if gate:
        ood = out_of_domain(symptom)
        if ood:
            return _declined(DECLINE_DOMAIN.format(ood), gate=True)
    scored = retriever.search_scored(symptom, k=k)
    if not scored:
        return _declined(DECLINE)
    top1 = scored[0][1]
    points = []
    transport_failures = []   # transport/HTTP errors, to tell UNAVAILABLE from DECLINE
    for i, (s, sc) in enumerate(scored, 1):
        tag = f"[S{i}]"
        user = (f"SYMPTOM: {symptom}\n\nSOURCE {tag}:\n{s.raw_text}\n\n"
                f"If relevant, give one candidate cause or next check from this source, ending "
                f"with {tag}. Otherwise output SKIP.")
        try:
            r = requests.post(f"{assert_local_endpoint(base_url).rstrip('/')}/api/chat", json={
                "model": model, "stream": False, "keep_alive": -1,
                "messages": [{"role": "system", "content": POINTWISE_SYSTEM},
                             {"role": "user", "content": user}],
                "options": {"temperature": 0, "seed": 0, "num_predict": 160},
            }, timeout=(10, 120))
            r.raise_for_status()          # a 4xx/5xx is a backend failure, not "no source"
            data = r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            transport_failures.append(repr(e)[:120])
            continue
        txt = ((data.get("message") or {}).get("content") or "").strip()
        if not txt or txt.upper().lstrip("*_ ").startswith("SKIP"):
            continue
        grounded = tag in txt
        if not grounded:
            # The model answered without echoing its source tag. The lead is KEPT and
            # attributed to the source it was generated from — that source was retrieved
            # and shown to the model, so the citation points at a real manual page — and
            # it is flagged `grounded: False`, rendered to the crew as
            # "(inferred from source — verify)". This is attribute-and-flag, NOT
            # cite-or-drop: nothing is discarded. `n_grounded` / `n_inferred` carry the
            # split.
            txt = f"{txt} {tag}"
        points.append({"text": txt, "tag": tag, "doc": s.source_doc, "score": round(sc, 1),
                       "grounded": grounded, "src_text": s.raw_text or "",
                       "snippet": " ".join((s.raw_text or "").split())[:120]})
    if not points:
        # Distinguish "the manuals do not cover this" from "the model was unreachable".
        # Previously every transport failure was reported as a coverage decline, which
        # blames the corpus for a backend outage.
        if transport_failures:
            out = _declined(f"{UNAVAILABLE} {transport_failures[0]})")
            out["unavailable"] = True
            return out
        return _declined(DECLINE)

    d = _dials(points, symptom, scored)
    n_grounded = sum(1 for p in points if p["grounded"])
    lines = []
    for j, p in enumerate(points, 1):
        marks = ""
        if not p["grounded"]:
            marks += "  (inferred from source — verify)"
        if p["weakly_sourced"]:
            marks += "  (weakly sourced — verify against source)"
        lines.append(f"{j}. {p['text']}{marks}")
    return {"declined": False, "diagnosis": "\n".join(lines),
            "cite_score": d["cite_score"], "coverage_flag": d["coverage_flag"],
            "groundedness": d["groundedness"], "support": d["support"],
            "consensus": d["consensus"], "consensus_term": d["consensus_term"],
            "agreement": d["agreement"], "retrieval_strength": round(top1, 1),
            "n_points": len(points), "n_grounded": n_grounded, "n_inferred": len(points) - n_grounded,
            "n_dropped_uncited": 0,
            "sources": [{"tag": p["tag"], "doc": p["doc"], "grounded": p["grounded"],
                         "extr": p["extr"], "weakly_sourced": p["weakly_sourced"],
                         "text": p["snippet"]} for p in points]}
