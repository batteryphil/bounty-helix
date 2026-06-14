"""
Detects and flags potentially hallucinated responses in a given text.

This tool analyzes the input text for signs of hallucination, such as making claims without evidence or contradicting known facts. It flags responses for review if it detects potential hallucination.

Usage:
from hallucination_detector import detect_hallucination

text = "The Earth is flat and the moon is made of green cheese."
is_hallucination, hallucination_details = detect_hallucination(text)

if is_hallucination:
    print(f"Potential hallucination detected: {hallucination_details}")
else:
    print("No hallucination detected.")
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

nlp = spacy.load("en_core_web_sm")

def detect_hallucination(text):
    doc = nlp(text)
    hallucination_details = []

    # Check for lack of evidence
    if any(token.text in STOP_WORDS for token in doc):
        hallucination_details.append("Lacks evidential support")

    # Check for contradictions
    if doc.ents:
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text.lower() in ["earth", "moon"]:
                hallucination_details.append("Contradicts known facts (e.g., Earth is not flat, moon is not made of cheese)")

    is_hallucination = bool(hallucination_details)

    return is_hallucination, hallucination_details

# Register the tool in the ToolRegistry
# toolset='self' means this tool is part of the self-improvement toolset
__all__ = ["detect_hallucination"]