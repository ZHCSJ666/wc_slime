import os
import json
import re
from typing import Dict, Any

# ----------------------------
# Config (can be overridden by env)
# ----------------------------

# Optional: if some samples do NOT have label, you can load English gold by qa_id
USE_EN_ANSWER_FILE = os.environ.get("WC_USE_EN_ANSWER_FILE", "0") == "1"

CHECK_ANSWER_DIR = os.environ.get(
    "WC_CHECK_ANSWER_DIR",
    "/export/home/pan/jiang_data/jiang/datasets/wc/mc_open_fenkai_check_answer",
)
EN_ANSWER_FILE = os.environ.get(
    "WC_EN_ANSWER_FILE",
    os.path.join(CHECK_ANSWER_DIR, "en_open.jsonl"),
)

# Correctness reward
REWARD_CORRECT = float(os.environ.get("WC_R_CORRECT", "1.0"))
PENALTY_WRONG = float(os.environ.get("WC_P_WRONG", "-0.3"))

# Format reward (only)
MAX_LEN_NO_PENALTY = int(os.environ.get("WC_MAX_LEN", "20"))
PENALTY_TOO_LONG = float(os.environ.get("WC_P_TOO_LONG", "-0.2"))
PENALTY_MULTILINE = float(os.environ.get("WC_P_MULTILINE", "-0.2"))

# Optional clipping (stability)
CLIP_MIN = float(os.environ.get("WC_CLIP_MIN", "-1.0"))
CLIP_MAX = float(os.environ.get("WC_CLIP_MAX", "1.2"))

# ----------------------------
# Normalization
# ----------------------------
_TRAIL_PUNCT_RE = re.compile(r"[ \t\r\n\.\,\!\?\;\:\，\。\！\？\；\：]+$")

def _norm_answer(s: str) -> str:
    """Light normalization: strip, first line, strip quotes/backticks, lowercase, trim trailing punctuation/spaces."""
    if s is None:
        return ""
    s = s.strip()
    if not s:
        return ""
    # Only first line (align with 'Print only the answer')
    s = s.splitlines()[0].strip()
    # Strip surrounding quotes/backticks
    s = s.strip("`'\"“”‘’")
    s = s.strip()
    # Remove trailing punctuation
    s = _TRAIL_PUNCT_RE.sub("", s)
    # Lowercase for robust matching
    return s.lower().strip()

# ----------------------------
# Optional: English answer index (qa_id -> normalized gold)
# ----------------------------
_en_answers: Dict[int, str] = {}
_loaded = False

def _load_en_answers_once() -> None:
    global _loaded, _en_answers
    if _loaded:
        return
    _loaded = True

    if not USE_EN_ANSWER_FILE:
        return
    if not os.path.exists(EN_ANSWER_FILE):
        return

    try:
        with open(EN_ANSWER_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                meta = obj.get("meta", {}) or {}
                qa_id = meta.get("qa_id", None)
                if qa_id is None:
                    continue

                msgs = obj.get("messages", []) or []
                gold = ""
                for m in reversed(msgs):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        gold = m.get("content", "") or ""
                        break
                _en_answers[int(qa_id)] = _norm_answer(gold)
    except Exception:
        # Fail-safe: if reading fails, just don't use it
        _en_answers = {}

# ----------------------------
# Format penalty
# ----------------------------
def _format_penalty(pred_raw: str) -> float:
    """Format reward: penalize multiline and too-long outputs."""
    if pred_raw is None:
        return 0.0

    r = 0.0
    # Multiline check (any newline)
    if "\n" in pred_raw or "\r" in pred_raw:
        r += PENALTY_MULTILINE

    t = pred_raw.strip()
    if len(t) > MAX_LEN_NO_PENALTY:
        r += PENALTY_TOO_LONG

    return float(r)

# ----------------------------
# Main reward
# ----------------------------
async def reward_func(args, sample, **kwargs) -> float:
    """
    Only two components:
    1) Correctness: match gold (prefer sample.label; optionally fallback to EN_ANSWER_FILE by qa_id)
    2) Format: multiline + length penalty
    """
    _load_en_answers_once()

    pred_raw = getattr(sample, "response", "") or ""
    pred = _norm_answer(pred_raw)

    # Gold: prefer label (your example data has it)
    gold = _norm_answer(getattr(sample, "label", "") or "")

    # Optional fallback: if label missing, use qa_id -> en answer
    if not gold and USE_EN_ANSWER_FILE:
        qa_id = None
        md: Any = getattr(sample, "metadata", None)
        if isinstance(md, dict):
            qa_id = md.get("qa_id", None)
        if qa_id is not None:
            gold = _en_answers.get(int(qa_id), "")

    # 1) Correctness
    if pred and gold and pred == gold:
        r_correct = REWARD_CORRECT
    else:
        r_correct = PENALTY_WRONG

    # 2) Format
    r_fmt = _format_penalty(pred_raw)

    r = float(r_correct + r_fmt)

    # Clip for stability
    if r > CLIP_MAX:
        r = CLIP_MAX
    if r < CLIP_MIN:
        r = CLIP_MIN
    return r