import re
from typing import List, Dict, Any, Optional

class AnswerVerifier:
    """
    Verification layer that executes post-generation grounding analysis:
    1. False-Abstention Protection: Overrides false 'not specified' answers if context contains the answer.
    2. Faithfulness & Hallucination Guardrail: Validates extracted entities against retrieved context.
    3. Abstention Precision: Ensures true missing queries abstain cleanly.
    """

    @classmethod
    def verify(
        cls,
        query: str,
        raw_answer: str,
        context_chunks: List[Dict[str, Any]],
        doc_metadata: Optional[Any] = None
    ) -> str:
        if not raw_answer or not raw_answer.strip():
            return "The document does not specify this information."

        ans_clean = raw_answer.strip()
        q_lower = query.lower().strip()
        is_abstention = any(p in ans_clean.lower() for p in [
            "does not specify", "not specified", "does not contain",
            "not mentioned", "not state", "no information"
        ])

        full_context = "\n\n".join([c.get("text", "") for c in context_chunks]) if context_chunks else ""
        if doc_metadata:
            first_p = getattr(doc_metadata, "first_page_text", "")
            if first_p and first_p not in full_context:
                full_context += f"\n\n{first_p}"

        ctx_lower = full_context.lower()

        # 1. FALSE-ABSTENTION RECOVERY
        if is_abstention and full_context:
            from app.rag.generator import LocalGenerator
            gen = LocalGenerator()
            
            # Check targeted attributes (Title, Experience, Contact, Name, Numbers)
            targeted_attr = gen._extract_targeted_attribute(query, full_context)
            if targeted_attr and len(targeted_attr) > 2 and "does not specify" not in targeted_attr.lower():
                return targeted_attr

            # Check keyword line match fallback
            stop_words = {"what", "whats", "who", "where", "when", "how", "why", "tell", "give", "this", "that", "with", "from", "the", "pdf", "doc", "document", "is", "are", "was", "were", "can", "she", "he", "they", "it", "name", "project", "explain", "ive", "please", "show", "me"}
            query_words = set([w.lower() for w in re.findall(r"\w+", query) if len(w) > 2 and w.lower() not in stop_words])

            if query_words:
                lines = [l.strip() for l in re.split(r"[\n\.;]+", full_context) if len(l.strip()) > 8]
                matched_lines = []
                for line in lines:
                    match_count = sum(1 for qw in query_words if qw in line.lower())
                    if match_count > 0:
                        matched_lines.append((match_count, line))

                if matched_lines:
                    matched_lines.sort(key=lambda x: x[0], reverse=True)
                    clean_match = re.sub(r'^(Cell\s*\d+\s*:|\*|\-)\s*', '', matched_lines[0][1], flags=re.I).strip()
                    return f"According to the document: {clean_match}."

        # 2. FAITHFULNESS & HALLUCINATION GUARDRAIL
        if not is_abstention and full_context:
            # Check numbers
            if any(k in q_lower for k in ["how many", "marks", "time", "date", "year", "page", "number"]):
                numbers_in_ans = re.findall(r'\b\d+\b', ans_clean)
                if numbers_in_ans and not any(num in ctx_lower for num in numbers_in_ans):
                    return "The document does not specify this information."

            # Check general key content words in answer to verify grounding in context
            ans_stop_words = {"the", "a", "an", "is", "are", "was", "were", "this", "that", "it", "according", "to", "document", "following", "lists"}
            ans_words = [w.lower() for w in re.findall(r"\w+", ans_clean) if len(w) > 2 and w.lower() not in ans_stop_words]
            if ans_words:
                matched_in_ctx = sum(1 for w in ans_words if w in ctx_lower)
                # If less than 20% of meaningful answer words exist in context, it's an ungrounded LLM hallucination
                if (float(matched_in_ctx) / float(len(ans_words))) < 0.2:
                    return "The document does not specify this information."

        return ans_clean
