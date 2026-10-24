"""Typed extraction from a label section.

Given the contraindications text of one label, return a list of distinct contraindications as a
pydantic model. The model sees only that section, must return JSON, and must not invent items.
Scored by eval/run_extract_eval.py against the section text itself (see there for what that
can and cannot prove).
"""
import json
import re

from pydantic import BaseModel, Field, ValidationError

from labellens import llm


class Contraindications(BaseModel):
    items: list[str] = Field(
        description="Each distinct contraindication as one short phrase, quoted or closely paraphrased from the text."
    )


SYSTEM_PROMPT = """You extract structured facts from one section of an FDA drug label.
Rules:
- Return ONLY a JSON object of the form {"items": ["...", "..."]}.
- One entry per distinct contraindication (a condition, population, or co-medication in which the drug must not be used).
- Use the label's own words. Do not add anything that is not in the text. Do not explain."""


def _parse(text: str) -> Contraindications:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return Contraindications.model_validate_json(text)
    except ValidationError:
        data = json.loads(text)  # model returned a bare list or a differently shaped object
        if isinstance(data, list):
            return Contraindications(items=[str(x) for x in data])
        raise


def extract_contraindications(section_text: str) -> tuple[Contraindications, llm.Completion]:
    completion = llm.chat(SYSTEM_PROMPT, f"Section text:\n\n{section_text}", json_mode=True)
    return _parse(completion.text), completion
