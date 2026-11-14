"""
ai_advisor.py
=============
"Sentinel Council" - a multi-LLM advisory layer for the The5ers trading bot.

Two independent counselors review each setup and their opinions are blended into a
single ``CouncilVerdict``:

* **Scout**        - Groq ``llama-3.3-70b-versatile``: technical momentum and chart
                     structure (trend, EMA slope, RSI posture, ATR context).
* **Risk Auditor** - Google Gemini ``gemini-2.5-flash``: The5ers prop-firm rule
                     compliance (-$100 daily stop, +$60 daily target, strict 1:3
                     minimum reward:risk floor).

Design contract
---------------
The council is **advisory only** and must never block, delay, or crash the trading
loop:

* every network call is bounded by a hard timeout (default ``DEFAULT_TIMEOUT`` = 8s);
* any failure - missing API key, SDK not installed, network error, malformed JSON -
  fails **open** with ``degraded=True`` and a neutral recommendation;
* ``deliberate()`` never raises.

Credentials are read from ``GROQ_API_KEY`` and ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``). If neither is present the council is simply inert.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("the5ers.council")

# A hard ceiling so a slow/hanging LLM can never stall a trading tick.
DEFAULT_TIMEOUT = 8.0

SCOUT_MODEL = "llama-3.3-70b-versatile"
AUDITOR_MODEL = "gemini-2.5-flash"

VALID_VERDICTS = {"bullish", "bearish", "neutral"}

_SCOUT_SYSTEM = (
    "You are Scout, a disciplined technical analyst on a multi-agent trading desk. "
    "Assess the momentum and chart structure of the described setup using the EMA "
    "relationship, RSI posture and ATR context provided. Be concise and decisive. "
    'Respond with ONLY a compact JSON object: {"verdict": "bullish"|"bearish"|"neutral", '
    '"confidence": 0.0-1.0, "summary": "one short sentence"}.'
)

_AUDITOR_SYSTEM = (
    "You are the Risk Auditor for a The5ers Bootcamp $5,000 prop-firm evaluation "
    "account. Hard rules: the daily loss limit is -$100, the daily profit target is "
    "+$60, and every trade must carry a minimum reward:risk of 1:3. Judge whether the "
    "proposed setup complies with these rules given the reported daily P&L and "
    "remaining risk budget. Be conservative and concise. "
    'Respond with ONLY a compact JSON object: {"verdict": "bullish"|"bearish"|"neutral", '
    '"confidence": 0.0-1.0, "compliant": true|false, "summary": "one short sentence"}.'
)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object found in ``text``."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip a ```json ... ``` fence if the model wrapped its answer.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_VERDICTS:
        return text
    if text.startswith("bull") or text in {"long", "buy", "up"}:
        return "bullish"
    if text.startswith("bear") or text in {"short", "sell", "down"}:
        return "bearish"
    return "neutral"


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if number > 1.0:  # tolerate a 0-100 style answer
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _format_context(context: Dict[str, Any]) -> str:
    """Render the setup context as a compact, model-friendly block."""
    lines = [f"- {key}: {value}" for key, value in context.items() if value is not None]
    return "Setup under review:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #
@dataclass
class CounselorOpinion:
    """A single counselor's parsed opinion."""

    role: str
    provider: str
    model: str
    verdict: str = "neutral"
    confidence: float = 0.0
    summary: str = ""
    compliant: Optional[bool] = None
    ok: bool = False
    error: str = ""
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CouncilVerdict:
    """The blended result of a council deliberation."""

    recommendation: str = "neutral"
    confidence: float = 0.0
    summary: str = ""
    scout: Optional[CounselorOpinion] = None
    auditor: Optional[CounselorOpinion] = None
    consensus: bool = False
    degraded: bool = True
    elapsed_ms: int = 0
    messages: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "consensus": self.consensus,
            "degraded": self.degraded,
            "elapsed_ms": self.elapsed_ms,
            "messages": list(self.messages),
            "timestamp": self.timestamp,
            "scout": self.scout.to_dict() if self.scout else None,
            "auditor": self.auditor.to_dict() if self.auditor else None,
        }


# --------------------------------------------------------------------------- #
# The council
# --------------------------------------------------------------------------- #
class SentinelCouncil:
    """Asynchronous multi-LLM advisory council. Fail-open by construction."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.timeout = float(timeout) if timeout and timeout > 0 else DEFAULT_TIMEOUT
        self._groq_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self._gemini_key = (
            gemini_api_key
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        ).strip()
        self.enabled = bool(enabled)
        self._groq_client: Any = None
        self._gemini_client: Any = None

    # ------------------------------------------------------------------ #
    # Introspection (dashboard)
    # ------------------------------------------------------------------ #
    @property
    def scout_available(self) -> bool:
        return bool(self.enabled and self._groq_key)

    @property
    def auditor_available(self) -> bool:
        return bool(self.enabled and self._gemini_key)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scout": self.scout_available,
            "auditor": self.auditor_available,
            "scout_model": SCOUT_MODEL,
            "auditor_model": AUDITOR_MODEL,
            "timeout": self.timeout,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def deliberate(self, context: Dict[str, Any]) -> CouncilVerdict:
        """Ask both counselors in parallel and blend their opinions.

        Never raises: every failure path returns a ``degraded`` verdict with a
        neutral recommendation so the caller can proceed unimpeded.
        """
        verdict = CouncilVerdict()
        if not self.enabled:
            verdict.messages.append("Council disabled.")
            return verdict

        coros: List[Any] = []
        labels: List[str] = []
        if self.scout_available:
            coros.append(self._call_scout(context))
            labels.append("scout")
        else:
            verdict.messages.append("Scout offline (no GROQ_API_KEY).")
        if self.auditor_available:
            coros.append(self._call_auditor(context))
            labels.append("auditor")
        else:
            verdict.messages.append("Risk Auditor offline (no GEMINI_API_KEY).")

        if not coros:
            verdict.messages.append("Council inert: no provider configured.")
            return verdict

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        except Exception as exc:  # pragma: no cover - gather itself shouldn't raise
            LOGGER.warning("Council gather failed open: %s", exc)
            verdict.messages.append(f"Council error: {exc}")
            return verdict

        for label, result in zip(labels, results):
            if isinstance(result, BaseException):
                LOGGER.warning("Counselor %s failed open: %s", label, result)
                verdict.messages.append(f"{label} error: {result}")
                continue
            if label == "scout":
                verdict.scout = result
            else:
                verdict.auditor = result

        self._blend(verdict)
        return verdict

    # ------------------------------------------------------------------ #
    # Blending
    # ------------------------------------------------------------------ #
    @staticmethod
    def _blend(verdict: CouncilVerdict) -> None:
        opinions = [o for o in (verdict.scout, verdict.auditor) if o is not None and o.ok]
        if not opinions:
            verdict.degraded = True
            verdict.recommendation = "neutral"
            verdict.confidence = 0.0
            verdict.summary = "Council unavailable - proceeding on the deterministic rules."
            return

        verdict.degraded = len(opinions) < 2
        # Risk dominates: prefer the auditor's view when both speak.
        primary = next((o for o in opinions if o.role == "Risk Auditor"), opinions[-1])
        verdict.recommendation = primary.verdict
        verdict.confidence = sum(o.confidence for o in opinions) / len(opinions)
        verdict.summary = primary.summary or opinions[0].summary

        votes = {o.verdict for o in opinions}
        verdict.consensus = len(votes) == 1
        if verdict.consensus:
            verdict.messages.append(f"Council consensus: {verdict.recommendation}.")
        else:
            verdict.messages.append("Council split - recommendation follows the Risk Auditor.")

    # ------------------------------------------------------------------ #
    # Counselors
    # ------------------------------------------------------------------ #
    async def _call_scout(self, context: Dict[str, Any]) -> CounselorOpinion:
        if self._groq_client is None:
            from groq import AsyncGroq  # lazy import keeps the module import-safe

            self._groq_client = AsyncGroq(api_key=self._groq_key)
        response = await asyncio.wait_for(
            self._groq_client.chat.completions.create(
                model=SCOUT_MODEL,
                messages=[
                    {"role": "system", "content": _SCOUT_SYSTEM},
                    {"role": "user", "content": _format_context(context)},
                ],
                temperature=0.2,
                max_tokens=400,
            ),
            timeout=self.timeout,
        )
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            text = ""
        return _opinion_from_text("Scout", "groq", SCOUT_MODEL, text)

    async def _call_auditor(self, context: Dict[str, Any]) -> CounselorOpinion:
        if self._gemini_client is None:
            from google import genai  # lazy import keeps the module import-safe

            self._gemini_client = genai.Client(api_key=self._gemini_key)
        response = await asyncio.wait_for(
            self._gemini_client.aio.models.generate_content(
                model=AUDITOR_MODEL,
                contents=f"{_AUDITOR_SYSTEM}\n\n{_format_context(context)}",
            ),
            timeout=self.timeout,
        )
        text = getattr(response, "text", "") or ""
        return _opinion_from_text("Risk Auditor", "gemini", AUDITOR_MODEL, text)


def _opinion_from_text(role: str, provider: str, model: str, text: str) -> CounselorOpinion:
    """Parse a counselor's raw text into a structured opinion (never raises)."""
    data = _extract_json(text)
    if data is None:
        return CounselorOpinion(
            role=role,
            provider=provider,
            model=model,
            verdict="neutral",
            confidence=0.5,
            summary=(text or "").strip()[:400] or "No usable response.",
            ok=True,
            raw=text or "",
        )
    compliant: Optional[bool] = None
    if "compliant" in data:
        compliant = bool(data.get("compliant"))
    return CounselorOpinion(
        role=role,
        provider=provider,
        model=model,
        verdict=_normalize_verdict(data.get("verdict")),
        confidence=_clamp_confidence(data.get("confidence")),
        summary=str(data.get("summary", "")).strip()[:400],
        compliant=compliant,
        ok=True,
        raw=text or "",
    )


__all__ = [
    "SentinelCouncil",
    "CouncilVerdict",
    "CounselorOpinion",
    "SCOUT_MODEL",
    "AUDITOR_MODEL",
    "DEFAULT_TIMEOUT",
]
