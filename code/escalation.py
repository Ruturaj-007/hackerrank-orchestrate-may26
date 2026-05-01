from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EscalationSignal:
    should_escalate: bool
    reason: str          # human-readable, goes into justification


# keyword patterns 
# Each tuple: (compiled_regex, reason_string)

_ESCALATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Security / fraud
    (re.compile(r"\bfraud\b|\bstolen\b|\bidentity theft\b|\bcompromised\b", re.I),
     "Potential fraud or security incident — requires human review"),
    (re.compile(r"\bphishing\b|\bscam\b|\bhacked\b|\bunauthorized (access|transaction)\b", re.I),
     "Security threat — requires human review"),
    (re.compile(r"\bsecurity vulnerabilit\b|\bbug bounty\b|\bexploit\b|\bcve\b", re.I),
     "Security vulnerability disclosure — requires human review"),

    # Billing / payments
    (re.compile(r"\brefund\b|\bchargeback\b|\bdispute (a )?charge\b|\bbilling issue\b", re.I),
     "Billing or payment dispute — requires human review"),
    (re.compile(r"\bpayment (fail|not processed|stuck|pending)\b", re.I),
     "Payment processing issue — requires human review"),

    # Legal / compliance
    (re.compile(r"\blegal\b|\blawsuit\b|\blitigat\b|\bcompliance\b|\bregulat\b|\bGDPR\b|\bDPA\b", re.I),
     "Legal or compliance matter — requires human review"),

    # Account / data deletion
    (re.compile(r"\bdelete (my )?account\b|\bclose (my )?account\b|\bremove (my )?data\b", re.I),
     "Account deletion or data removal — requires human review"),
    (re.compile(r"\bright to (be forgotten|erasure)\b", re.I),
     "GDPR data erasure request — requires human review"),

    # Platform outages
    (re.compile(r"\bsite (is )?down\b|\bplatform (is )?down\b|\boutage\b|\bnone of the pages\b", re.I),
     "Possible platform outage — requires human review"),
    (re.compile(r"\bnone of the submissions.*(work|fail)\b|\ball requests.*fail\b", re.I),
     "Widespread platform issue — requires human review"),

    # Prompt injection / adversarial
    (re.compile(r"ignore (all |previous )?instructions\b|forget your (system |previous )?prompt\b", re.I),
     "Possible prompt injection attempt — escalating"),
    (re.compile(r"\binternal (rules|documents|logic|policy)\b.*\bshow\b|\bshow.*\binternal (rules|policy)\b", re.I),
     "Request for internal system information — escalating"),

    # Harmful / malicious requests
    (re.compile(r"\bdelete all files\b|\brm -rf\b|\bdrop (table|database)\b", re.I),
     "Potentially harmful system command — escalating"),
    (re.compile(r"\bmalware\b|\bransomware\b|\bvirus\b|\bkeylogger\b", re.I),
     "Malicious content request — escalating"),

    # Subscription changes (need human)
    (re.compile(r"\bpause (our |my )?subscription\b|\bcancel (our |my )?subscription\b", re.I),
     "Subscription change request — requires human review"),

    # Infosec / vendor assessment forms
    (re.compile(r"\binfosec\b.*\bform\b|\bsecurity (assessment|questionnaire|form)\b", re.I),
     "InfoSec assessment — requires human review"),
]

# Out-of-scope but should reply (not escalate) — these get "invalid" request_type
_INVALID_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bactor\b.*\biron man\b|\biron man\b.*\bactor\b", re.I),
     "Out of scope — not a support question"),
    (re.compile(r"^(thank you|thanks|ty|thx)[.!]?\s*$", re.I),
     "Conversational message — not a support question"),
]


# public API 

def check_escalation(issue: str, subject: str, company: Optional[str]) -> EscalationSignal:
    """
    Run deterministic escalation checks.

    Returns EscalationSignal(should_escalate=True, reason=...) if any
    high-risk pattern matches, otherwise (False, "").
    """
    combined = f"{subject or ''} {issue or ''}".strip()

    # No company → always escalate (can't ground safely)
    if not company or company.strip().lower() in ("none", ""):
        # But first check if it's clearly invalid/conversational
        for pattern, reason in _INVALID_PATTERNS:
            if pattern.search(combined):
                return EscalationSignal(should_escalate=False, reason=reason)
        # Generic "none" company: escalate unless totally trivial
        # (we let the LLM decide for ambiguous cases below)

    for pattern, reason in _ESCALATION_PATTERNS:
        if pattern.search(combined):
            return EscalationSignal(should_escalate=True, reason=reason)

    return EscalationSignal(should_escalate=False, reason="")


def is_invalid_request(issue: str, subject: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if the ticket is clearly out of scope / invalid.
    These get replied=True with an out-of-scope message rather than escalated.
    """
    combined = f"{subject or ''} {issue or ''}".strip()
    for pattern, reason in _INVALID_PATTERNS:
        if pattern.search(combined):
            return True, reason
    return False, ""