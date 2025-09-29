from __future__ import annotations

import re
from typing import Dict

import numpy as np


class ProductionPGDExtractor:
    """
    Fast, regex-first extractor (spaCy optional). Returns a stable 35D Riemannian vector
    plus a small Banach vector attached separately by the caller if desired.
    """
    def __init__(self) -> None:
        self._r_keys = [f"syn_{i}" for i in range(6)] + \
                       [f"math_{i}" for i in range(8)] + \
                       [f"code_{i}" for i in range(7)] + \
                       [f"sem_{i}" for i in range(6)] + \
                       [f"aux_{i}" for i in range(8)]  # pad to 35 if some groups are light

    def extract_features(self, prompt: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        # syntactic (cheap proxies)
        sents = [s for s in re.split(r"[.!?]\s+", prompt) if s]
        feats["syn_0"] = float(np.mean([len(s.split()) for s in sents])) if sents else 0.0
        feats["syn_1"] = float(np.std([len(s.split()) for s in sents])) if sents else 0.0
        feats["syn_2"] = float(len(sents))
        feats["syn_3"] = float(prompt.count(","))
        feats["syn_4"] = float(prompt.count(";"))
        feats["syn_5"] = float(min(len(prompt), 4096))  # proxy for length

        # math
        math_ops = len(re.findall(r"[∑∏∫∂∇≤≥≠≈]|\\(sum|int|prod|frac|cdot)", prompt))
        latex = len(re.findall(r"\$[^$]+\$|\\begin{equation}", prompt))
        feats |= {
            "math_0": math_ops,
            "math_1": latex,
            "math_2": float(len(re.findall(r"\bprove|derive|compute|solve\b", prompt, re.I))),
            "math_3": float(len(re.findall(r"[0-9]+(\\.[0-9]+)?", prompt))),
            "math_4": float(prompt.count("^")+prompt.count("_")),
            "math_5": float("theorem" in prompt.lower()),
            "math_6": float("lemma" in prompt.lower()),
            "math_7": float("proof" in prompt.lower()),
        }

        # code
        code_blocks = len(re.findall(r"```[\s\S]*?```", prompt))
        lang_hits = len(re.findall(r"\b(python|sql|javascript|cpp|java|rust|go)\b", prompt, re.I))
        feats |= {
            "code_0": float(code_blocks),
            "code_1": float(lang_hits),
            "code_2": float(
                len(re.findall(r"\bfor|while|if|else|try|catch|except\b", prompt, re.I))
            ),
            "code_3": float(len(re.findall(r"[{}();]", prompt))),
            "code_4": float("class " in prompt or "def " in prompt),
            "code_5": float("SELECT " in prompt.upper()),
            "code_6": float("import " in prompt),
        }

        # semantic proxies
        tokens = prompt.split()
        diffs = [
            abs(len(tokens[i + 1]) - len(tokens[i])) for i in range(len(tokens) - 1)
        ] if len(tokens) > 1 else []
        feats |= {
            "sem_0": float(np.sum(diffs)) if diffs else 0.0,
            "sem_1": float(np.mean(diffs)) if diffs else 0.0,
            "sem_2": float(np.std(diffs)) if diffs else 0.0,
            "sem_3": float(len(set([t.lower() for t in tokens]))),
            "sem_4": float(len(tokens)),
            "sem_5": float(len(set(w for w in tokens if len(w)>6))),
        }

        # aux padding (zeros)
        for i in range(8):
            feats[f"aux_{i}"] = feats.get(f"aux_{i}", 0.0)

        # minimal Banach (pragmatic) features for demo
        feats["prag_latency_class"] = 1.0
        feats["prag_cost_class"] = 1.0
        feats["prag_pii_level"] = 0.0
        feats["prag_region_eu_only"] = 0.0
        return feats
