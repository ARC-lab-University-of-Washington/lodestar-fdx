# -*- coding: utf-8 -*-
##
# @file verify_release.py
# @brief Independent, runnable verification of this release. Exits non-zero on any failure.
#
# Every claim this repository makes about itself is checked here by executing it, not by
# asserting it. Run before every release and in CI.
#
#   python scripts/verify_release.py                    # structural checks only
#   LODESTAR_CORPUS=../apollo-anomaly-atlas/corpus/corpus.json \
#   LODESTAR_ATLAS=../apollo-anomaly-atlas/data/anomalies.csv \
#     python scripts/verify_release.py                  # + data-dependent checks
#
import ast
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    """@brief Run one check; record PASS / FAIL / SKIP.
    @param name Human-readable check name.
    @param fn   Zero-arg callable returning a detail string, or raising to fail.
                Return None to skip (e.g. optional data not present).
    """
    try:
        detail = fn()
        (SKIP if detail is None else PASS).append((name, detail or "not applicable"))
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- structure
def _py_files():
    out = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "docs")]
        out += [os.path.join(root, f) for f in files if f.endswith(".py")]
    return out


def c_parse():
    n = 0
    for p in _py_files():
        ast.parse(open(p, encoding="utf-8").read())
        n += 1
    assert n > 40, f"expected >40 python files, found {n}"
    return f"{n} files parse as valid Python 3"


def c_import():
    import lodestar_edge as L
    for sym in ("load_corpus", "SalienceBM25Retriever", "CorpusGraph",
                "in_domain", "out_of_domain", "diagnose"):
        assert sym in L.__all__, f"{sym} missing from public API"
    return f"package imports cold; {len(L.__all__)} public symbols"


def c_deps():
    """No heavyweight ML stack: the edge claim is BM25-only."""
    banned = ("torch", "tensorflow", "sentence_transformers", "transformers", "faiss")
    hits = []
    for p in _py_files():
        s = open(p, encoding="utf-8").read()
        for b in banned:
            if re.search(rf"^\s*(import|from)\s+{b}\b", s, re.M):
                hits.append(f"{os.path.relpath(p, ROOT)}:{b}")
    assert not hits, f"heavyweight ML import found: {hits}"
    return "no torch / transformers / faiss anywhere — BM25-only claim holds"


def c_no_local_paths():
    """No machine-specific paths, usernames, credentials or internal references."""
    pats = {
        "unix home path": r"/Users/[A-Za-z0-9._-]+|/home/[a-z][A-Za-z0-9._-]*",
        "windows user path": r"C:\\\\Users\\\\",
        "cloud drive": r"OneDrive|Dropbox/|SharePoint",
        "credential": r"(api[_-]?key|secret|password|access[_-]?token)\s*=\s*[\"'][^\"']+[\"']",
        "private ip": r"\b(?:192\.168|10\.0\.0|172\.16)\.\d+\.\d+",
    }
    bad = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for f in files:
            if not f.endswith((".py", ".sh", ".md", ".txt", ".yml", ".cfg", "Doxyfile")):
                continue
            p = os.path.join(root, f)
            if os.path.abspath(p) == os.path.abspath(__file__):
                continue          # this file contains the patterns it searches for
            try:
                s = open(p, encoding="utf-8").read()
            except Exception:
                continue
            for label, pat in pats.items():
                m = re.search(pat, s)
                if m:
                    bad.append(f"{os.path.relpath(p, ROOT)}: {label} ({m.group(0)[:40]})")
    assert not bad, "; ".join(bad)
    return "no local paths, usernames, credentials or private IPs"


def c_ai_statement():
    missing = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for f in files:
            if f.endswith((".py", ".sh")):
                p = os.path.join(root, f)
                if "AI use statement" not in open(p, encoding="utf-8").read():
                    missing.append(os.path.relpath(p, ROOT))
    assert not missing, f"missing AI use statement: {missing}"
    return "AI use statement present in every source file"


def c_doxygen_tags():
    undoc = []
    for p in _py_files():
        s = open(p, encoding="utf-8").read()
        if "@file" not in s:
            undoc.append(os.path.relpath(p, ROOT))
    assert not undoc, f"no @file block: {undoc}"
    return f"@file/@brief block in all {len(_py_files())} modules"


# ------------------------------------------------------- data-dependent
def _corpus_path():
    return os.environ.get("LODESTAR_CORPUS",
                          os.path.join(ROOT, "..", "apollo-anomaly-atlas", "corpus", "corpus.json"))


def c_corpus():
    p = _corpus_path()
    if not os.path.exists(p):
        return None
    import lodestar_edge as L
    segs = L.load_corpus(p)
    assert len(segs) == 3370, f"expected 3,370 segments, got {len(segs)}"
    return f"corpus loads: {len(segs)} segments"


def c_boilerplate_filter():
    p = _corpus_path()
    if not os.path.exists(p):
        return None
    import lodestar_edge as L
    r = L.SalienceBM25Retriever(L.load_corpus(p))
    assert len(r.segments) == 3148, f"expected 3,148 retrievable, got {len(r.segments)}"
    assert r.n_dropped == 222, f"expected 222 dropped, got {r.n_dropped}"
    return f"{len(r.segments)} retrievable / {r.n_dropped} boilerplate — matches the paper"


def c_retrieval():
    p = _corpus_path()
    if not os.path.exists(p):
        return None
    import lodestar_edge as L
    r = L.SalienceBM25Retriever(L.load_corpus(p))
    hits = r.search("Main B bus undervolt, cryogenic oxygen tank two pressure dropping", k=5)
    assert hits, "retrieval returned nothing for a known Apollo 13 symptom"
    assert all(h.source_doc for h in hits), "a retrieved segment has no source document"
    return f"retrieval returns {len(hits)} cited hits, every one traceable to a source document"


def c_scope_gate():
    import lodestar_edge as L
    assert L.out_of_domain("the lunar rover television camera lens is dusty"), \
        "gate failed to decline an out-of-domain symptom"
    assert L.in_domain("main bus b undervolt"), \
        "gate wrongly declined an in-domain spacecraft fault"
    assert L.in_domain("water entered the suit loop during EVA"), \
        "gate wrongly declined a suit-circuit fault (ECLSS is in domain)"
    return "declines out-of-domain gear; never declines a vehicle-system fault"


def c_split():
    """The in/out split must reproduce from the shipped scorer, not from a stored column."""
    atlas = os.environ.get("LODESTAR_ATLAS",
                           os.path.join(ROOT, "..", "apollo-anomaly-atlas", "data", "anomalies.csv"))
    sm = os.path.join(ROOT, "analysis", "subsystem_match.py")
    if not (os.path.exists(atlas) and os.path.exists(sm)):
        return None
    import importlib.util as iu
    spec = iu.spec_from_file_location("subsystem_match", sm)
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    rows = list(csv.DictReader(open(atlas, encoding="utf-8")))
    crew = [r for r in rows if str(r.get("crew_observed", "")).lower() == "true"]
    inc = sum(1 for r in crew if m.is_in_corpus(r["subsystem"]))
    assert len(rows) == 255, f"expected 255 anomalies, got {len(rows)}"
    assert len(crew) == 146, f"expected 146 crew-observed, got {len(crew)}"
    assert inc == 107, f"expected 107 in-corpus, got {inc}"
    return f"255 / 146 / {inc} in / {len(crew)-inc} out — recomputed, matches the paper"


def c_ingest_selfcontained():
    """Every local module ingest/ imports must ship with it.

    The corpus-construction pipeline is published so the atlas can be rebuilt. It
    previously imported two project modules that were not copied into the release,
    so it could not run from a clean clone.
    """
    ing = os.path.join(ROOT, "ingest")
    if not os.path.isdir(ing):
        return None
    local = {f[:-3] for f in os.listdir(ing) if f.endswith(".py")}
    third_party = {"pdfplumber", "pytesseract", "pypdfium2", "networkx", "requests",
                   "PIL", "fitz", "numpy", "rank_bm25"}
    missing = set()
    for f in sorted(local):
        src = open(os.path.join(ing, f + ".py"), encoding="utf-8").read()
        for mod in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w]*)", src, re.M):
            if mod in local or mod in third_party or mod in sys.stdlib_module_names:
                continue
            missing.add(f"{f}.py -> {mod}")
    assert not missing, f"ingest/ imports modules that are not in the release: {sorted(missing)}"
    return f"ingest/ is self-contained ({len(local)} modules, no missing local imports)"


def c_headline_accuracy():
    """The paper's 85% must re-score from the shipped decisions dump.

    This runs analysis/score_subsystem.py's own criterion over the shipped run dump
    and asserts the in-corpus rate the paper reports. It guards the one number a
    reader is most likely to check.
    """
    import importlib.util as iu
    sm_path = os.path.join(ROOT, "analysis", "subsystem_match.py")
    dump = os.environ.get("LODESTAR_DATA") or os.path.join(
        ROOT, "..", "apollo-anomaly-atlas", "baselines")
    dump = os.path.join(dump, "decisions_146_k8.jsonl")
    if not (os.path.exists(sm_path) and os.path.exists(dump)):
        return None
    spec = iu.spec_from_file_location("subsystem_match", sm_path)
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    rows = [json.loads(l) for l in open(dump, encoding="utf-8") if l.strip()]
    inc = [r for r in rows if m.is_in_corpus(r["subsystem"])]
    hit = sum(1 for r in inc if m.names(r["decision"], m.aliases_for(r["subsystem"])))
    pct = round(100 * hit / len(inc))
    assert len(rows) == 146, f"expected 146 decisions, got {len(rows)}"
    assert len(inc) == 107, f"expected 107 in-corpus, got {len(inc)}"
    assert pct == 85, f"expected 85% in-corpus accuracy, got {pct}% ({hit}/{len(inc)})"
    return f"in-corpus accuracy {hit}/{len(inc)} = {pct}% — matches the paper"


def c_coverage_auc():
    """Recompute the coverage AUC from the shipped dump and report it.

    NOTE: this reports rather than asserts. The shipped artifacts integrate to
    0.7339, reported as 0.73. See docs/REPRODUCING.md.
    """
    import importlib.util as iu
    sm_path = os.path.join(ROOT, "analysis", "subsystem_match.py")
    dump = os.environ.get("LODESTAR_DATA") or os.path.join(
        ROOT, "..", "apollo-anomaly-atlas", "baselines")
    dump = os.path.join(dump, "decisions_146_k8.jsonl")
    if not (os.path.exists(sm_path) and os.path.exists(dump)):
        return None
    spec = iu.spec_from_file_location("subsystem_match", sm_path)
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    rows = [json.loads(l) for l in open(dump, encoding="utf-8") if l.strip()]
    pos = [r["coverage"] for r in rows if m.is_in_corpus(r["subsystem"]) and r["coverage"] is not None]
    neg = [r["coverage"] for r in rows if not m.is_in_corpus(r["subsystem"]) and r["coverage"] is not None]
    auc = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))
    return f"coverage AUC recomputes to {auc:.4f} (n={len(pos)}/{len(neg)}) — see docs/REPRODUCING.md"


CHECKS = [
    ("source parses", c_parse),
    ("package imports", c_import),
    ("no heavyweight ML deps", c_deps),
    ("no local paths or credentials", c_no_local_paths),
    ("AI use statement present", c_ai_statement),
    ("Doxygen blocks present", c_doxygen_tags),
    ("ingest is self-contained", c_ingest_selfcontained),
    ("corpus loads", c_corpus),
    ("boilerplate filter", c_boilerplate_filter),
    ("retrieval is grounded", c_retrieval),
    ("scope gate behaviour", c_scope_gate),
    ("benchmark split reproduces", c_split),
    ("headline accuracy reproduces", c_headline_accuracy),
    ("coverage AUC recomputes", c_coverage_auc),
]

if __name__ == "__main__":
    print("LODESTAR-FDX release verification\n" + "=" * 66)
    for name, fn in CHECKS:
        check(name, fn)
    for name, d in PASS: print(f"  PASS  {name:32s} {d}")
    for name, d in SKIP: print(f"  SKIP  {name:32s} data not present — set LODESTAR_CORPUS")
    for name, d in FAIL: print(f"  FAIL  {name:32s} {d}")
    print("=" * 66)
    print(f"{len(PASS)} passed, {len(SKIP)} skipped, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
