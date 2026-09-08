import re
from enum import Enum
from typing import Dict, Any

class QueryIntent(str, Enum):
    METADATA = "metadata"
    SUMMARY = "summary"
    LIST = "list"
    COMPARISON = "comparison"
    FACT = "fact"

class QueryClassifier:
    METADATA_PATTERNS = [
        r"\b(what's|whats|what\s+is)\s+(the\s+)?(title|name|subject|course)\b",
        r"\btitle\s+of\s+(this|the|a|my)?\s*(pdf|doc|document|file|txt)\b",
        r"\bwhat\s+is\s+(this|the)\s+(pdf|doc|document|file)\s+called\b",
        r"\b(document|pdf|file)\s+title\b",
        r"\b(how\s+many|total)\s+pages\b",
        r"\bpage\s+count\b",
        r"\b(what's|whats|what\s+is)\s+(the\s+)?file\s*name\b",
        r"\bwho\s+(wrote|authored|created)\s+this\b",
        r"\bauthor\s+of\b"
    ]

    SUMMARY_PATTERNS = [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\bwhat\s+is\s+this\s+(doc|pdf|document|txt|file)\s+about\b",
        r"\bwhat\s+is\s+about\b",
        r"\bexplain\s+(this\s+)?(document|pdf|file)\b",
        r"\bgive\s+me\s+an?\s+overview\b",
        r"\btell\s+me\s+about\s+this\s+(document|pdf|file)\b",
        r"\bmain\s+topics\b"
    ]

    LIST_PATTERNS = [
        r"\blist\s+(all|every)\b",
        r"\bshow\s+(all|every)\b",
        r"\blist\s+(the\s+)?(sections|topics|causes|symptoms|items|points|features)\b",
        r"\bwhat\s+are\s+(all\s+)?the\s+(sections|topics|causes|steps|modules)\b"
    ]

    COMPARISON_PATTERNS = [
        r"\bcompare\b",
        r"\bdifference\s+between\b",
        r"\bversus\b",
        r"\b\w+\s+vs\.?\s+\w+\b"
    ]

    @classmethod
    def classify(cls, query: str) -> QueryIntent:
        q_lower = query.strip().lower()

        # Check metadata queries first
        for pat in cls.METADATA_PATTERNS:
            if re.search(pat, q_lower):
                return QueryIntent.METADATA

        # Check summary queries
        for pat in cls.SUMMARY_PATTERNS:
            if re.search(pat, q_lower):
                return QueryIntent.SUMMARY

        # Check list queries
        for pat in cls.LIST_PATTERNS:
            if re.search(pat, q_lower):
                return QueryIntent.LIST

        # Check comparison queries
        for pat in cls.COMPARISON_PATTERNS:
            if re.search(pat, q_lower):
                return QueryIntent.COMPARISON

        # Default is pinpoint FACT QA
        return QueryIntent.FACT
