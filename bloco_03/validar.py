"""
Match Validator
───────────────
Receives the full pipeline context (intent classification + DB match)
and produces a trust score + structured reasoning.

The script (agent orchestrator) uses the score to decide:
  score >= EXECUTE_THRESHOLD  → execute command
  score >= CONFIRM_THRESHOLD  → ask user to confirm
  score <  CONFIRM_THRESHOLD  → reject / ask clarification

Output schema:
{
  "trust_score":  float,        # 0.0 – 1.0
  "decision":     str,          # "EXECUTE" | "CONFIRM" | "REJECT"
  "signals": [
    {
      "name":    str,           # signal identifier
      "weight":  float,         # contribution to final score
      "value":   float,         # raw measured value
      "status":  str,           # "ok" | "warn" | "fail"
      "reason":  str            # human-readable explanation
    }
  ],
  "summary": str                # one-line explanation for logs / UI
}
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal

# ─────────────────────────────────────────────
# THRESHOLDS  (tune these per environment)
# ─────────────────────────────────────────────
EXECUTE_THRESHOLD = 0.65   # auto-execute above this
CONFIRM_THRESHOLD = 0.40   # ask user between this and EXECUTE


# ─────────────────────────────────────────────
# SIGNAL WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHTS = {
    "intent_confidence":   0.30,   # how sure the classifier was
    "vector_distance":     0.30,   # how close the DB match is
    "rank_coherence":      0.20,   # does command complexity match intent rank?
    "language_coherence":  0.10,   # input language vs command language
    "category_command_fit":0.10,   # does command category fit intent label?
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ─────────────────────────────────────────────
# SIGNAL EVALUATION HELPERS
# ─────────────────────────────────────────────

# Expected DB categories for each intent label
INTENT_CATEGORY_MAP: dict[str, set[str]] = {
    "SEARCH":   {"filesystem", "docker", "process", "network", "git", "database", "search"},
    "CREATE":   {"filesystem", "git", "docker", "code", "scaffold", "database"},
    "QUESTION": {"system", "network", "process", "docker", "filesystem", "git"},
    "UNKNOWN":  set(),  # anything is suspicious
}

# Commands that are destructive / high-risk (raise confirmation bar)
DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(rm|rmdir|drop|truncate|delete|kill|format|mkfs|dd|shutdown|reboot|"
    r"chmod\s+777|chown\s+root|sudo\s+rm)\b",
    re.IGNORECASE,
)

# Rough token sets for language detection (pt vs en)
PT_TOKENS = {"listar", "criar", "buscar", "mostrar", "verificar", "executar",
             "remover", "atualizar", "instalar", "configurar", "onde", "qual",
             "quais", "como", "porque", "quando", "quanto"}
EN_TOKENS = {"list", "create", "find", "show", "check", "run", "remove",
             "update", "install", "configure", "where", "which", "how",
             "why", "when", "what"}


def _detect_language(text: str) -> Literal["pt", "en", "unknown"]:
    words = set(text.lower().split())
    pt_hits = len(words & PT_TOKENS)
    en_hits = len(words & EN_TOKENS)
    if pt_hits > en_hits:
        return "pt"
    if en_hits > pt_hits:
        return "en"
    return "unknown"


def _command_language(command: str) -> Literal["en", "unknown"]:
    """Shell commands are always English; returns 'en' if it looks like a command."""
    cmd = command.strip().split()[0] if command.strip() else ""
    # Most Unix commands are 2-10 lowercase ASCII chars
    if re.match(r"^[a-z]{1,15}$", cmd):
        return "en"
    return "unknown"


def _estimate_command_complexity(command: str) -> int:
    """
    Rough complexity rank (1-5) for a shell command based on:
    - pipe count
    - argument count
    - subshell / redirection presence
    """
    pipes      = command.count("|")
    args       = len(command.split()) - 1
    subshell   = int("$(" in command or "`" in command)
    redirect   = int(">" in command or "<" in command)
    flags      = len(re.findall(r"\s-{1,2}\w+", command))

    raw = 1 + pipes + (args // 3) + subshell + redirect + (flags // 2)
    return max(1, min(5, raw))


# ─────────────────────────────────────────────
# SIGNAL EVALUATORS
# ─────────────────────────────────────────────

@dataclass
class Signal:
    name:   str
    weight: float
    value:  float         # 0.0 – 1.0
    status: str           # "ok" | "warn" | "fail"
    reason: str

    @property
    def contribution(self) -> float:
        return self.weight * self.value


def _sig_intent_confidence(intent: dict) -> Signal:
    conf = float(intent.get("confidence", 0))
    if conf >= 0.60:
        status, reason = "ok",   f"Classifier confidence {conf:.2f} is strong"
    elif conf >= 0.35:
        status, reason = "warn", f"Classifier confidence {conf:.2f} is moderate"
    else:
        status, reason = "fail", f"Classifier confidence {conf:.2f} is low — intent unclear"
    return Signal("intent_confidence", WEIGHTS["intent_confidence"], conf, status, reason)


def _sig_vector_distance(ctx: dict) -> Signal:
    """
    Vector distance from DB: lower = better match.
    Typical cosine distance range: 0 (identical) → 2 (opposite).
    We invert and normalize to [0, 1] score.
    """
    dist = float(ctx.get("vector_search_distance", 1.0))
    # Score: 1.0 at dist=0, 0.0 at dist>=1.2
    score = max(0.0, 1.0 - dist / 1.2)
    if score >= 0.60:
        status, reason = "ok",   f"DB match distance {dist:.3f} — close semantic match"
    elif score >= 0.30:
        status, reason = "warn", f"DB match distance {dist:.3f} — partial match"
    else:
        status, reason = "fail", f"DB match distance {dist:.3f} — poor match, command may be wrong"
    return Signal("vector_distance", WEIGHTS["vector_distance"], score, status, reason)


def _sig_rank_coherence(intent: dict, ctx: dict) -> Signal:
    """
    The complexity rank of the retrieved command should be close
    to the intent rank. A rank-1 intent getting a rank-5 command is suspicious.
    """
    intent_rank  = int(intent.get("rank", 3))
    command_rank = _estimate_command_complexity(ctx.get("comando", ""))
    delta        = abs(intent_rank - command_rank)

    if delta == 0:
        score, status, reason = 1.0, "ok",   f"Rank coherent: intent={intent_rank}, command≈{command_rank}"
    elif delta == 1:
        score, status, reason = 0.75, "ok",  f"Rank close: intent={intent_rank}, command≈{command_rank}"
    elif delta == 2:
        score, status, reason = 0.45, "warn", f"Rank mismatch: intent={intent_rank}, command≈{command_rank}"
    else:
        score, status, reason = 0.10, "fail", f"Large rank gap: intent={intent_rank}, command≈{command_rank} — complexity mismatch"
    return Signal("rank_coherence", WEIGHTS["rank_coherence"], score, status, reason)


def _sig_language_coherence(raw_input: str, ctx: dict) -> Signal:
    """
    Input may be in Portuguese; the retrieved command should still be
    a valid shell command (always English). If input is PT but the
    command looks like natural language text, something went wrong.
    """
    input_lang   = _detect_language(raw_input)
    command_lang = _command_language(ctx.get("comando", ""))

    if command_lang == "en":
        # Shell commands are language-agnostic; PT input → EN command is fine
        score  = 1.0
        status = "ok"
        reason = f"Input is '{input_lang}', command is a valid shell command (language-neutral)"
    else:
        score  = 0.30
        status = "warn"
        reason = f"Command '{ctx.get('comando','')}' doesn't look like a shell command"

    return Signal("language_coherence", WEIGHTS["language_coherence"], score, status, reason)


def _sig_category_command_fit(intent: dict, ctx: dict) -> Signal:
    """
    The DB category tag should be compatible with the intent label.
    E.g. a SEARCH intent should not resolve to a 'scaffold' category.
    """
    label    = intent.get("label", "UNKNOWN")
    db_cat   = ctx.get("categoria", "").lower()
    allowed  = INTENT_CATEGORY_MAP.get(label, set())

    if label == "UNKNOWN":
        score, status, reason = 0.10, "fail", "Intent is UNKNOWN — cannot validate category fit"
    elif not db_cat:
        score, status, reason = 0.50, "warn", "No category tag in DB record"
    elif db_cat in allowed:
        score, status, reason = 1.0,  "ok",   f"Category '{db_cat}' fits intent '{label}'"
    else:
        score, status, reason = 0.20, "fail", f"Category '{db_cat}' unexpected for intent '{label}'"

    return Signal("category_command_fit", WEIGHTS["category_command_fit"], score, status, reason)


# ─────────────────────────────────────────────
# DESTRUCTIVE COMMAND PENALTY
# ─────────────────────────────────────────────

def _destructive_penalty(command: str) -> float:
    """Returns a multiplier to downgrade trust if command is destructive."""
    if DESTRUCTIVE_PATTERNS.search(command):
        return 0.60   # cap trust at 60% for destructive commands
    return 1.0


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def validate(pipeline_output: dict) -> dict:
    """
    Main entry point.

    Args:
        pipeline_output: the full JSON produced by the pipeline, e.g.
        {
          "entrada_original":     "listar diretorios",
          "classificacao_intent": {"label": "SEARCH", "rank": 1, "confidence": 0.32},
          "contexto_banco_dados": {"comando": "docker images", "categoria": "docker",
                                   "descricao": "...", "vector_search_distance": 0.94}
        }

    Returns:
        Validation result dict (see module docstring for schema).
    """
    raw_input = pipeline_output.get("entrada_original", "")
    intent    = pipeline_output.get("classificacao_intent", {})
    ctx       = pipeline_output.get("contexto_banco_dados", {})
    command   = ctx.get("comando", "")

    # Evaluate every signal
    signals: list[Signal] = [
        _sig_intent_confidence(intent),
        _sig_vector_distance(ctx),
        _sig_rank_coherence(intent, ctx),
        _sig_language_coherence(raw_input, ctx),
        _sig_category_command_fit(intent, ctx),
    ]

    # Weighted sum → raw trust score
    raw_score = sum(s.contribution for s in signals)

    # Apply destructive penalty (multiplicative cap)
    penalty    = _destructive_penalty(command)
    trust_score = round(min(1.0, raw_score * penalty), 4)

    # Decision gate
    if trust_score >= EXECUTE_THRESHOLD:
        decision = "EXECUTE"
    elif trust_score >= CONFIRM_THRESHOLD:
        decision = "CONFIRM"
    else:
        decision = "REJECT"

    # Summary
    fail_reasons  = [s.reason for s in signals if s.status == "fail"]
    warn_reasons  = [s.reason for s in signals if s.status == "warn"]
    if decision == "EXECUTE":
        summary = f"High trust ({trust_score:.2f}) — safe to run '{command}'"
    elif decision == "CONFIRM":
        issues  = (fail_reasons + warn_reasons)[:2]
        summary = f"Moderate trust ({trust_score:.2f}) — confirm before running '{command}'. Issues: {'; '.join(issues)}"
    else:
        issues  = (fail_reasons + warn_reasons)[:2]
        summary = f"Low trust ({trust_score:.2f}) — rejected '{command}'. Reasons: {'; '.join(issues)}"

    return {
        "trust_score": trust_score,
        "decision":    decision,
        "signals": [
            {
                "name":         s.name,
                "weight":       s.weight,
                "value":        round(s.value, 4),
                "contribution": round(s.contribution, 4),
                "status":       s.status,
                "reason":       s.reason,
            }
            for s in signals
        ],
        "summary":        summary,
        "destructive":    penalty < 1.0,
        "thresholds":     {"execute": EXECUTE_THRESHOLD, "confirm": CONFIRM_THRESHOLD},
    }