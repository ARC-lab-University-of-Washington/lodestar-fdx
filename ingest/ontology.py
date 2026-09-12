# AI use statement: docs/AI_USE.md
##
# @file ontology.py
# @brief Fixed ontology for Lodestar's fault knowledge graph — mirrors HybridRAG's approach (5 entity types / 8 relation types / attributes, defined before ext.
#
# raction) rather than open-ended entity discovery. A closed schema is what
# makes triples mergeable across chunks and gives the LLM extractor a
# constrained, checkable target instead of free text.
#
"""
Fixed ontology for Lodestar's fault knowledge graph — mirrors HybridRAG's
approach (5 entity types / 8 relation types / attributes, defined before
extraction) rather than open-ended entity discovery. A closed schema is
what makes triples mergeable across chunks and gives the LLM extractor a
constrained, checkable target instead of free text.
"""

ENTITY_TYPES = [
    "Symptom",       # observed anomaly, e.g. "low cabin O2 partial pressure"
    "Subsystem",     # e.g. "ECLSS", "Thermal"
    "Component",     # e.g. "SOV-204", "pump P-2"
    "Fault",         # named failure mode, e.g. "valve fails to seat"
    "Procedure",     # a citable segment id (malfunction procedure / check / step)
]

RELATION_TYPES = [
    "indicates",       # Symptom -> Fault
    "locatedIn",        # Component -> Subsystem
    "hasComponent",      # Subsystem -> Component
    "causes",          # Fault -> Fault (causal chain)
    "diagnosedBy",      # Fault -> Procedure (check that confirms it)
    "resolvedBy",        # Fault -> Procedure (corrective step)
    "references",       # Procedure -> Procedure (cross-ref, "see para X")
    "precedes",          # Procedure -> Procedure (ordering within a flow)
]

TRIPLE_EXTRACTION_PROMPT = """You extract structured triples from spacecraft engineering text.
Use ONLY these entity types: {entity_types}
Use ONLY these relation types: {relation_types}

Return strict JSON: {{"triples": [{{"subject": str, "subject_type": str,
"relation": str, "object": str, "object_type": str}}, ...]}}

Rules:
- Only extract what the text explicitly states — no inference beyond the text.
- Entity names should be the exact engineering terms/IDs used in the text
  (component IDs, symptom phrases, subsystem names).
- If the text contains none of the schema's relations, return {{"triples": []}}.

TEXT:
{text}
"""
