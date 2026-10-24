"""Typed extraction from a label section.

Given the text of one section of one label, return a list of distinct items as a pydantic model.
The model sees only that section, must return JSON, and must not invent items. Each supported
section has its own one-line instruction saying what counts as an item; the schema, parsing and
scoring are shared. Scored by eval/run_extract_eval.py against the section text itself (see there
for what that can and cannot prove).
"""
import json
import re

from pydantic import BaseModel, Field, ValidationError

from labellens import llm


class Items(BaseModel):
    items: list[str] = Field(description="Each distinct item as one short phrase, quoted or closely paraphrased from the text.")


Contraindications = Items  # kept for callers that name the type

# What one item is, per section. Adding a section is one line here.
ITEM_RULES = {
    "contraindications": "One entry per distinct contraindication (a condition, population, or co-medication in which the drug must not be used).",
    "drug_interactions": "One entry per interacting drug or drug class, followed by the effect or action the label describes for it.",
    "warnings_and_cautions": "One entry per distinct warning or precaution (a risk the label tells prescribers to watch for).",
}

SYSTEM_PROMPT = """You extract structured facts from one section of an FDA drug label.
Rules:
- Return ONLY a JSON object of the form {{"items": ["...", "..."]}}.
- {rule}
- Use the label's own words. Do not add anything that is not in the text. Do not explain."""


def _as_text(item) -> str:
    """Small models sometimes return {"drug": "...", "effect": "..."} instead of one string."""
    if isinstance(item, dict):
        return ": ".join(str(v) for v in item.values() if v not in (None, ""))
    return str(item)


def _parse(text: str) -> Items:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return Items.model_validate_json(text)
    except ValidationError:
        data = json.loads(text)  # bare list, or items that are objects rather than strings
        items = data if isinstance(data, list) else data.get("items")
        if isinstance(items, list):
            return Items(items=[_as_text(x) for x in items])
        raise


def extract_items(section: str, section_text: str) -> tuple[Items, llm.Completion]:
    if section not in ITEM_RULES:
        raise ValueError(f"no extraction rule for section {section!r}; known: {sorted(ITEM_RULES)}")
    system = SYSTEM_PROMPT.format(rule=ITEM_RULES[section])
    completion = llm.chat(system, f"Section text:\n\n{section_text}", json_mode=True)
    return _parse(completion.text), completion


def extract_contraindications(section_text: str) -> tuple[Items, llm.Completion]:
    return extract_items("contraindications", section_text)
