from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from classifier import classify, generate_response, escalation_response
from escalation import check_escalation, is_invalid_request
from rag import Retriever, TOP_K_DEFAULT


# output type 

@dataclass
class TicketResult:
    status:        str           # "replied" | "escalated"
    product_area:  str
    response:      str
    justification: str
    request_type:  str           # "product_issue"|"feature_request"|"bug"|"invalid"
    sources:       list[str] = field(default_factory=list)
    escalation_rule: str = ""
    elapsed_s: float = 0.0


# agent

class TriageAgent:
    """
    Multi-domain support triage agent.

    Args:
        retriever: pre-built Retriever over the corpus
        k:         number of chunks to retrieve per query
    """

    def __init__(self, retriever: Retriever, k: int = TOP_K_DEFAULT):
        self._retriever = retriever
        self._k         = k

    # public 

    def process(
        self,
        issue:   str,
        subject: str,
        company: Optional[str],
    ) -> TicketResult:
        """Process one support ticket end-to-end."""
        t0 = time.time()

        issue   = (issue   or "").strip()
        subject = (subject or "").strip()
        company_clean = (company or "").strip() or None
        if company_clean and company_clean.lower() == "none":
            company_clean = None

        # 0. Trivial / invalid check 
        invalid, invalid_reason = is_invalid_request(issue, subject)
        if invalid:
            return TicketResult(
                status        = "replied",
                product_area  = "conversation_management",
                response      = "I'm sorry, this is out of scope from my capabilities.",
                justification = invalid_reason,
                request_type  = "invalid",
                elapsed_s     = time.time() - t0,
            )

        # 1. Deterministic escalation gate 
        escalation_signal = check_escalation(issue, subject, company_clean)

        # 2. RAG retrieval 
        query   = f"{subject} {issue}".strip()
        results = self._retriever.retrieve(query, company=company_clean, k=self._k)
        context = self._retriever.format_context(results)
        sources = [r[0].source_file for r in results]

        #  3. LLM classification 
        try:
            cls = classify(issue, subject, company_clean, context)
        except Exception as exc:
            # LLM failure → safe fallback: escalate
            return TicketResult(
                status        = "escalated",
                product_area  = "general_support",
                response      = escalation_response(str(exc), company_clean),
                justification = f"LLM classification failed: {exc}",
                request_type  = "product_issue",
                sources       = sources,
                elapsed_s     = time.time() - t0,
            )

        # 4. Merge escalation signals 
        #   Deterministic rule wins over LLM if it says escalate.
        #   LLM can additionally escalate cases the rules miss.
        final_escalate = escalation_signal.should_escalate or cls.get("should_escalate", False)
        escalation_reason = escalation_signal.reason or cls.get("escalation_reason", "")

        # 5. Generate response or escalation message 
        if final_escalate:
            response = escalation_response(escalation_reason, company_clean)
            status   = "escalated"
            justification = (
                cls.get("justification", "")
                or escalation_reason
                or "Escalated due to sensitive or high-risk content."
            )
        else:
            try:
                response = generate_response(
                    issue, subject, company_clean, context, cls["product_area"]
                )
            except Exception as exc:
                response = escalation_response(str(exc), company_clean)
                status   = "escalated"
                return TicketResult(
                    status        = status,
                    product_area  = cls.get("product_area", "general_support"),
                    response      = response,
                    justification = f"Response generation failed: {exc}",
                    request_type  = cls.get("request_type", "product_issue"),
                    sources       = sources,
                    escalation_rule = escalation_reason,
                    elapsed_s     = time.time() - t0,
                )
            status        = "replied"
            justification = cls.get("justification", "Answered from support corpus.")

        return TicketResult(
            status          = status,
            product_area    = cls.get("product_area", "general_support"),
            response        = response,
            justification   = justification,
            request_type    = cls.get("request_type", "product_issue"),
            sources         = sources,
            escalation_rule = escalation_signal.reason,
            elapsed_s       = time.time() - t0,
        )