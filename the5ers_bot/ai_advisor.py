"""
ai_advisor.py
=============
"Sentinel Council" - a multi-LLM advisory layer for the The5ers trading bot.

Two independent counselors review each setup and their opinions are blended into a
single ``CouncilVerdict``:

* **Scout**        - Groq: technical momentum and chart structure (trend, EMA
                     slope, RSI posture, ATR context).
* **Risk Auditor** - Google Gemini: The5ers prop-firm rule compliance (-$100 daily
                     stop, +$60 daily target, strict 1:3 minimum reward:risk floor).

Model cascade
-------------
Each counselor walks a prioritised list of models and moves to the next one on any
API error, rate-limit or timeout:

* Scout (Groq)        : ``openai/gpt-oss-120b`` -> ``qwen/qwen3.8-27b`` -> ``openai/gpt-oss-20b``
* Risk Auditor (Gemini): ``gemini-2.5-flash`` -> ``gemini-2.5-flash-lite`` -> ``gemini-3.8-flash``

The whole cascade shares a single hard deadline (``DEFAULT_TIMEOUT`` = 8s), so a
deliberation can never overrun and block the trading loop. Which model actually
answered is recorded on the verdict (``scout_model_used`` / ``auditor_model_used``)
plus a ``fallback_used`` flag for dashboard telemetry.

Design contract
---------------
The council is **advisory only** and must never block, delay, or crash the trading
loop:

* every network call is bounded by the shared hard timeout;
* any failure - missing API key, SDK not installed, network error, malformed JSON -
  fails **open** with ``approved=True``, ``degraded=True`` and a neutral read;
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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("the5ers.council")

# A hard ceiling so a slow/hanging LLM can never stall a trading tick. The whole
# model cascade for a counselor shares this single deadline.
DEFAULT_TIMEOUT = 8.0

# Groq Scout cascade (in order of preference).
SCOUT_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]
# Gemini Risk Auditor cascade (in order of preference).
AUDITOR_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
]

# Primary models, kept for display / backward compatibility.
SCOUT_MODEL = SCOUT_MODELS[0]
AUDITOR_MODEL = AUDITOR_MODELS[0]

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
    # True when the primary model failed and a fallback model answered.
    fallback: bool = False
    # Every model the cascade attempted, in order (telemetry).
    attempted: List[str] = field(default_factory=list)

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
    # Fail-open by default: the council only ever advises, never blocks.
    approved: bool = True
    # Which exact model answered for each provider (empty when the provider failed).
    scout_model_used: str = ""
    auditor_model_used: str = ""
    fallback_used: bool = False
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
            "approved": self.approved,
            "scout_model_used": self.scout_model_used,
            "auditor_model_used": self.auditor_model_used,
            "fallback_used": self.fallback_used,
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
    """Asynchronous multi-LLM advisory council with model fallback. Fail-open."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        enabled: bool = True,
        scout_models: Optional[Sequence[str]] = None,
        auditor_models: Optional[Sequence[str]] = None,
    ) -> None:
        self.timeout = float(timeout) if timeout and timeout > 0 else DEFAULT_TIMEOUT
        self._groq_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self._gemini_key = (
            gemini_api_key
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        ).strip()
        self.enabled = bool(enabled)
        self.scout_models: List[str] = list(scout_models) if scout_models else list(SCOUT_MODELS)
        self.auditor_models: List[str] = list(auditor_models) if auditor_models else list(AUDITOR_MODELS)
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
            "scout_model": self.scout_models[0],
            "auditor_model": self.auditor_models[0],
            "scout_models": list(self.scout_models),
            "auditor_models": list(self.auditor_models),
            "timeout": self.timeout,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def deliberate(self, context: Dict[str, Any]) -> CouncilVerdict:
        """Ask both counselors in parallel and blend their opinions.

        Never raises: every failure path returns a ``degraded``, ``approved``
        verdict with a neutral recommendation so the caller can proceed.
        """
        started = time.monotonic()
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
            verdict.messages.append("Scout offline (no Groq API key).")
        if self.auditor_available:
            coros.append(self._call_auditor(context))
            labels.append("auditor")
        else:
            verdict.messages.append("Risk Auditor offline (no Gemini API key).")

        if not coros:
            verdict.messages.append("Council inert: no provider configured.")
            self._blend(verdict)
            verdict.elapsed_ms = int((time.monotonic() - started) * 1000)
            return verdict

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        except Exception as exc:  # pragma: no cover - gather itself shouldn't raise
            LOGGER.warning("Council gather failed open: %s", exc)
            verdict.messages.append(f"Council error: {exc}")
            self._blend(verdict)
            verdict.elapsed_ms = int((time.monotonic() - started) * 1000)
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
            if not result.ok:
                verdict.messages.append(f"{label} unavailable: {result.error or 'unknown error'}")

        self._blend(verdict)
        verdict.elapsed_ms = int((time.monotonic() - started) * 1000)
        return verdict

    # ------------------------------------------------------------------ #
    # Blending
    # ------------------------------------------------------------------ #
    @staticmethod
    def _blend(verdict: CouncilVerdict) -> None:
        if verdict.scout is not None and verdict.scout.ok:
            verdict.scout_model_used = verdict.scout.model
        if verdict.auditor is not None and verdict.auditor.ok:
            verdict.auditor_model_used = verdict.auditor.model

        opinions = [o for o in (verdict.scout, verdict.auditor) if o is not None and o.ok]
        if not opinions:
            # Fail-open: both providers failed (or only one was configured and it
            # failed) - the deterministic rules still govern trading.
            verdict.degraded = True
            verdict.approved = True
            verdict.recommendation = "neutral"
            verdict.confidence = 0.0
            verdict.summary = "Council unavailable - proceeding on the deterministic rules."
            return

        verdict.degraded = len(opinions) < 2
        verdict.fallback_used = any(o.fallback for o in opinions)
        for opinion in opinions:
            if opinion.fallback:
                verdict.messages.append(
                    f"{opinion.role}: primary model failed, fallback '{opinion.model}' answered."
                )

        # Risk dominates: prefer the auditor's view when both speak.
        primary = next((o for o in opinions if o.role == "Risk Auditor"), opinions[-1])
        verdict.recommendation = primary.verdict
        verdict.confidence = sum(o.confidence for o in opinions) / len(opinions)
        verdict.summary = primary.summary or opinions[0].summary

        # Fail-open: only an explicit, auditor-checked "no" revokes approval.
        auditor = verdict.auditor
        if auditor is not None and auditor.ok and auditor.compliant is False:
            verdict.approved = False
            verdict.messages.append("Risk Auditor flagged a compliance breach.")
        else:
            verdict.approved = True

        if verdict.scout_model_used:
            verdict.messages.append(f"Scout answered with '{verdict.scout_model_used}'.")
        if verdict.auditor_model_used:
            verdict.messages.append(f"Risk Auditor answered with '{verdict.auditor_model_used}'.")

        votes = {o.verdict for o in opinions}
        verdict.consensus = len(votes) == 1
        if verdict.consensus:
            verdict.messages.append(f"Council consensus: {verdict.recommendation}.")
        else:
            verdict.messages.append("Council split - recommendation follows the Risk Auditor.")

    # ------------------------------------------------------------------ #
    # Counselors (model cascades)
    # ------------------------------------------------------------------ #
    async def _call_scout(self, context: Dict[str, Any]) -> CounselorOpinion:
        return await self._run_cascade(
            role="Scout",
            provider="groq",
            models=self.scout_models,
            make_call=lambda model: self._scout_once(model, context),
        )

    async def _call_auditor(self, context: Dict[str, Any]) -> CounselorOpinion:
        return await self._run_cascade(
            role="Risk Auditor",
            provider="gemini",
            models=self.auditor_models,
            make_call=lambda model: self._auditor_once(model, context),
        )

    async def _run_cascade(
        self,
        role: str,
        provider: str,
        models: Sequence[str],
        make_call: Callable[[str], Awaitable[CounselorOpinion]],
    ) -> CounselorOpinion:
        """Try each model in ``models`` in order under a single hard deadline.

        The entire cascade shares ``self.timeout`` seconds: each attempt gets an
        equal slice of the remaining budget, so a slow primary still leaves the
        fallbacks a chance while the total can never overrun the deadline. Never
        raises - returns a failed ``CounselorOpinion`` when every model fails.
        """
        cascade = list(models) or [""]
        deadline = time.monotonic() + self.timeout
        attempted: List[str] = []
        last_error = ""

        for index, model in enumerate(cascade):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = last_error or "time budget exhausted"
                break
            # Give every remaining model an equal slice so the whole cascade is
            # attempted and still fits inside the shared deadline.
            budget = remaining / max(1, len(cascade) - index)
            attempted.append(model)
            try:
                opinion = await asyncio.wait_for(make_call(model), timeout=budget)
            except asyncio.TimeoutError:
                last_error = f"{model}: timed out after {budget:.1f}s"
                LOGGER.warning("%s model '%s' timed out; trying next fallback.", role, model)
                continue
            except Exception as exc:  # noqa: BLE001 - any provider error falls through
                last_error = f"{model}: {exc}"
                LOGGER.warning("%s model '%s' failed (%s); trying next fallback.", role, model, exc)
                continue
            if not isinstance(opinion, CounselorOpinion):
                last_error = f"{model}: malformed counselor response"
                continue

            opinion.model = model
            opinion.attempted = list(attempted)
            opinion.fallback = index > 0
            if index > 0:
                LOGGER.info("%s recovered on fallback model '%s'.", role, model)
            return opinion

        return CounselorOpinion(
            role=role,
            provider=provider,
            model=cascade[0],
            ok=False,
            error=last_error or "no model available",
            attempted=list(attempted),
        )

    async def _scout_once(self, model: str, context: Dict[str, Any]) -> CounselorOpinion:
        if self._groq_client is None:
            from groq import AsyncGroq  # lazy import keeps the module import-safe

            self._groq_client = AsyncGroq(api_key=self._groq_key)
        response = await self._groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SCOUT_SYSTEM},
                {"role": "user", "content": _format_context(context)},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            text = ""
        return _opinion_from_text("Scout", "groq", model, text)

    async def _auditor_once(self, model: str, context: Dict[str, Any]) -> CounselorOpinion:
        if self._gemini_client is None:
            from google import genai  # lazy import keeps the module import-safe

            self._gemini_client = genai.Client(api_key=self._gemini_key)
        response = await self._gemini_client.aio.models.generate_content(
            model=model,
            contents=f"{_AUDITOR_SYSTEM}\n\n{_format_context(context)}",
        )
        text = getattr(response, "text", "") or ""
        return _opinion_from_text("Risk Auditor", "gemini", model, text)


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
    "SCOUT_MODELS",
    "AUDITOR_MODELS",
    "DEFAULT_TIMEOUT",
]
