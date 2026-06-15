import logging
import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

ENTITIES_TO_ANONYMIZE = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",
    "MEDICAL_LICENSE",
]

# Single words that spaCy incorrectly tags as PERSON in document context
PERSON_FALSE_POSITIVES = {
    "email", "phone", "ssn", "id", "page", "policy", "employee",
    "manager", "director", "officer", "chief", "head",
}


def _build_analyzer() -> AnalyzerEngine:
    analyzer = AnalyzerEngine()

    # Custom recognizer: names in bullet-list format
    bullet_name_pattern = Pattern(
        name="bullet_list_name_pattern",
        regex=r"\b[A-Z][a-z]+\s[A-Z][a-z]+(?=,\s*Employee)",
        score=0.85,
    )
    bullet_name_recognizer = PatternRecognizer(
        supported_entity="PERSON",
        patterns=[bullet_name_pattern],
        name="BulletListNameRecognizer",
    )

    # Custom recognizer: SSN pattern (NNN-NN-NNNN)
    # Presidio's built-in US_SSN requires surrounding context words
    # This pattern catches bare SSN numbers regardless of context
    ssn_pattern = Pattern(
        name="ssn_pattern",
        regex=r"\b\d{3}-\d{2}-\d{4}\b",
        score=0.85,
    )
    ssn_recognizer = PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[ssn_pattern],
        name="SSNRecognizer",
    )

    analyzer.registry.add_recognizer(bullet_name_recognizer)
    analyzer.registry.add_recognizer(ssn_recognizer)
    return analyzer


class PIIScrubber:
    def __init__(self, score_threshold: float = 0.35):
        self.score_threshold = score_threshold
        logger.info("Initializing Presidio Analyzer and Anonymizer...")
        self.analyzer = _build_analyzer()
        self.anonymizer = AnonymizerEngine()
        logger.info("Presidio initialized successfully")

    def _filter_false_positives(
        self,
        results: list,
        text: str
    ) -> list:
        """
        Remove known false positives from analyzer results.
        Specifically: single-word PERSON detections that are
        common document keywords, not actual names.
        """
        filtered = []
        for r in results:
            if r.entity_type == "PERSON":
                matched_text = text[r.start:r.end].strip()
                # Keep only multi-word PERSON detections
                # Single words are almost always false positives
                # in enterprise document context
                if (
                    len(matched_text.split()) < 2
                    or matched_text.lower() in PERSON_FALSE_POSITIVES
                ):
                    logger.debug(f"Filtered false positive PERSON: '{matched_text}'")
                    continue
            filtered.append(r)
        return filtered

    def scrub(self, text: str) -> tuple[str, list[dict]]:
        if not text or not text.strip():
            return text, []

        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=ENTITIES_TO_ANONYMIZE,
            language="en",
            score_threshold=self.score_threshold,
        )

        if not analyzer_results:
            return text, []

        # Filter false positives before anonymizing
        analyzer_results = self._filter_false_positives(analyzer_results, text)

        if not analyzer_results:
            return text, []

        detected_entities = [
            {
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": round(result.score, 3),
                "original_value": text[result.start:result.end],
            }
            for result in analyzer_results
        ]

        logger.info(
            f"Detected {len(detected_entities)} PII entities: "
            f"{[e['entity_type'] for e in detected_entities]}"
        )

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                for entity in ENTITIES_TO_ANONYMIZE
            },
        )

        return anonymized_result.text, detected_entities