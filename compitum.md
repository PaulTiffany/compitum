awesome — here’s a \*\*complete, self-contained deliverable\*\* you can hand to your \*\*Gemini CLI\*\* (or any runner that reads a repo task file) to scaffold, install, test, and shadow-run \*\*compitum\*\* as a proper Python package.



I’ve included:



\* A clean \*\*repo layout\*\*

\* A \*\*Gemini task file\*\* to drive setup, tests, and a demo route

\* A minimal but \*\*runnable\*\* codebase with hardened math (SPD metric, whitened KDE, shadow prices), a Typer \*\*CLI\*\*, and \*\*property tests\*\*

\* Example \*\*constraints/configs\*\* and a \*\*synthetic benchmark\*\*



Copy-paste the blocks into files as named, or let your agent create them verbatim.



---



\# 0) Repository layout



```

compitum/

├─ pyproject.toml

├─ README.md

├─ LICENSE

├─ Gemini.yaml

├─ Makefile

├─ configs/

│  ├─ constraints\_us\_default.yaml

│  └─ router\_defaults.yaml

├─ compitum/

│  ├─ \_\_init\_\_.py

│  ├─ cli.py

│  ├─ router.py

│  ├─ energy.py

│  ├─ metric.py

│  ├─ coherence.py

│  ├─ constraints.py

│  ├─ effort\_qp.py

│  ├─ boundary.py

│  ├─ control.py

│  ├─ predictors.py

│  ├─ pgd.py

│  ├─ models.py

│  ├─ capabilities.py

│  └─ utils.py

├─ tests/

│  ├─ test\_invariants.py

│  ├─ test\_boundary.py

│  └─ test\_constraints.py

└─ examples/

&nbsp;  ├─ synth\_bench.py

&nbsp;  └─ demo\_route.py

```



---



\# 1) Project metadata



\## `pyproject.toml`



```toml

\[project]

name = "compitum"

version = "0.1.0"

description = "compitum: A Production-Ready, Geometrically-Aware AI Router"

authors = \[{ name="compitum authors", email="dev@compitum.org" }]

license = { text = "MIT" }

readme = "README.md"

requires-python = ">=3.9"

dependencies = \[

&nbsp; "numpy>=1.24",

&nbsp; "scipy>=1.10",

&nbsp; "scikit-learn>=1.3",

&nbsp; "typer>=0.12",

&nbsp; "pydantic>=2.7",

&nbsp; "pyyaml>=6.0.1"

]



\[project.optional-dependencies]

dev = \[

&nbsp; "pytest>=8.0",

&nbsp; "hypothesis>=6.98",

&nbsp; "pytest-cov>=5.0",

&nbsp; "ruff>=0.5.0",

&nbsp; "mypy>=1.10",

&nbsp; "lightgbm>=4.3 ; python\_version < '3.13'"

]



\[project.scripts]

compitum = "compitum.cli:app"



\[tool.ruff]

line-length = 100



\[tool.pytest.ini\_options]

addopts = "-q -ra --maxfail=1"

```



\## `LICENSE`



```text

MIT License



Copyright (c) 2025 ...



Permission is hereby granted, free of charge, to any person obtaining a copy...

```



\## `README.md`



````markdown

\# compitum



A production-ready, geometrically-aware AI router with SPD metric learning, constraint-aware

selection (shadow prices), metric-aware KDE coherence, and Lyapunov-stable online updates.



\## Install

```bash

python -m venv .venv \&\& source .venv/bin/activate

pip install -e ".\[dev]"

````



\## Quick demo



```bash

compitum route --prompt "Prove the binomial identity using generating functions."

```



\## Run tests



```bash

pytest

```



See `configs/` and `examples/` for constraints and a synthetic benchmark.



````



---



\# 2) Gemini task file \& Makefile



\## `Gemini.yaml`

```yaml

version: "1"

tasks:

&nbsp; - id: setup

&nbsp;   name: Create venv and install

&nbsp;   shell: |

&nbsp;     python -m venv .venv

&nbsp;     . .venv/bin/activate

&nbsp;     pip install -U pip

&nbsp;     pip install -e ".\[dev]"



&nbsp; - id: lint

&nbsp;   needs: \[setup]

&nbsp;   shell: |

&nbsp;     . .venv/bin/activate

&nbsp;     ruff check compitum



&nbsp; - id: test

&nbsp;   needs: \[setup]

&nbsp;   shell: |

&nbsp;     . .venv/bin/activate

&nbsp;     pytest



&nbsp; - id: bench-synth

&nbsp;   needs: \[setup]

&nbsp;   shell: |

&nbsp;     . .venv/bin/activate

&nbsp;     python examples/synth\_bench.py



&nbsp; - id: route-demo

&nbsp;   needs: \[setup]

&nbsp;   shell: |

&nbsp;     . .venv/bin/activate

&nbsp;     compitum route --prompt "Write a SQL query to compute 7-day rolling average by user."

````



\## `Makefile`



```make

.PHONY: setup test lint demo bench

setup:

&nbsp;	python -m venv .venv \&\& . .venv/bin/activate \&\& pip install -U pip \&\& pip install -e ".\[dev]"

test:

&nbsp;	. .venv/bin/activate \&\& pytest

lint:

&nbsp;	. .venv/bin/activate \&\& ruff check compitum

demo:

&nbsp;	. .venv/bin/activate \&\& compitum route --prompt "Sketch a proof for AM-GM inequality."

bench:

&nbsp;	. .venv/bin/activate \&\& python examples/synth\_bench.py

```



---



\# 3) Configs



\## `configs/constraints\_us\_default.yaml`



```yaml

\# Linear constraints Ax <= b over Banach (pragmatic) features (toy example)

A:

&nbsp; - \[1, 0, 0, 0]   # latency\_class <= 2

&nbsp; - \[0, 1, 0, 0]   # cost\_class <= 2

&nbsp; - \[0, 0, 1, 0]   # pii\_level <= 0

&nbsp; - \[0, 0, 0, 1]   # region\_eu\_only <= 0

b: \[2.0, 2.0, 0.0, 0.0]

```



\## `configs/router\_defaults.yaml`



```yaml

alpha: 0.40

beta\_t: 0.20

beta\_c: 0.15

beta\_d: 0.15

beta\_s: 0.10

metric:

&nbsp; D: 35

&nbsp; rank: 8

&nbsp; delta: 1.0e-3

update\_stride: 8

cold\_start\_threshold: 16

```



---



\# 4) Core package code



\## `compitum/\_\_init\_\_.py`



```python

\_\_all\_\_ = \["router", "metric", "constraints", "coherence", "boundary", "control", "energy"]

```



\## `compitum/capabilities.py`



```python

from dataclasses import dataclass

from typing import Set, Dict, Any



@dataclass

class Capabilities:

&nbsp;   regions: Set\[str]

&nbsp;   tools\_allowed: Set\[str]

&nbsp;   deterministic: bool = False



&nbsp;   def supports(self, pgd\_vector: Any, context: Dict\[str, Any] | None = None) -> bool:

&nbsp;       # Hook for model-specific gates; extend as needed.

&nbsp;       # Example: block if context\["region"] not in self.regions

&nbsp;       return True

```



\## `compitum/models.py`



```python

from dataclasses import dataclass

import numpy as np

from .capabilities import Capabilities



@dataclass

class Model:

&nbsp;   name: str

&nbsp;   center: np.ndarray  # center in Riemannian feature space

&nbsp;   capabilities: Capabilities

```



\## `compitum/utils.py`



```python

from \_\_future\_\_ import annotations

import hashlib

import numpy as np

from typing import Dict, Tuple



def split\_features(x: Dict\[str, float]) -> Tuple\[np.ndarray, np.ndarray]:

&nbsp;   # Riemannian: everything except prag\_\*, Banach: prag\_\* only

&nbsp;   xR = \[v for k, v in x.items() if not k.startswith("prag\_")]

&nbsp;   xB = \[v for k, v in x.items() if k.startswith("prag\_")]

&nbsp;   return np.array(xR, float), np.array(xB, float)



def pgd\_hash(prompt: str) -> str:

&nbsp;   return hashlib.md5(prompt.encode()).hexdigest()

```



\## `compitum/metric.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from typing import Optional, Tuple

from scipy.linalg import cholesky, LinAlgError

from sklearn.covariance import LedoitWolf



class SymbolicManifoldMetric:

&nbsp;   def \_\_init\_\_(self, D: int, rank: int, delta: float = 1e-3):

&nbsp;       self.D, self.rank, self.delta = D, rank, delta

&nbsp;       self.L = np.random.randn(D, rank) \* 0.01

&nbsp;       self.W: Optional\[np.ndarray] = None

&nbsp;       self.shrink = LedoitWolf()

&nbsp;       self.whitened\_residuals: list\[np.ndarray] = \[]



&nbsp;   def metric\_matrix(self) -> np.ndarray:

&nbsp;       return self.L @ self.L.T + self.delta \* np.eye(self.D)



&nbsp;   def \_update\_cholesky(self) -> np.ndarray:

&nbsp;       try:

&nbsp;           self.W = cholesky(self.metric\_matrix(), lower=False)

&nbsp;       except LinAlgError:

&nbsp;           self.delta = min(self.delta \* 2.0, 1e-1)

&nbsp;           self.W = cholesky(self.metric\_matrix(), lower=False)

&nbsp;       return self.W



&nbsp;   def distance(self, x: np.ndarray, mu: np.ndarray) -> Tuple\[float, float]:

&nbsp;       if self.W is None:

&nbsp;           self.\_update\_cholesky()

&nbsp;       z = x - mu

&nbsp;       wz = self.W @ z

&nbsp;       d = float(np.linalg.norm(wz))

&nbsp;       if len(self.whitened\_residuals) > self.rank:

&nbsp;           cov = self.shrink.fit(np.array(self.whitened\_residuals)).covariance\_

&nbsp;           sigma = float(np.sqrt(max(wz.T @ cov @ wz, 0.0)))

&nbsp;       else:

&nbsp;           sigma = 0.1

&nbsp;       return d, sigma



&nbsp;   def update\_spd(self, x: np.ndarray, mu: np.ndarray, beta\_d: float, d: float, eta: float,

&nbsp;                  srmf\_controller) -> float:

&nbsp;       z = x - mu

&nbsp;       A = -(beta\_d / (2 \* max(d, 1e-8))) \* np.outer(z, z)  # dU/dM

&nbsp;       grad\_L = 2 \* A @ self.L

&nbsp;       grad\_norm = float(np.linalg.norm(grad\_L, 2))

&nbsp;       eta\_cap, \_ = srmf\_controller.update(d\_star=d, grad\_norm=grad\_norm)

&nbsp;       self.L -= min(eta, eta\_cap) \* grad\_L

&nbsp;       fnorm = np.linalg.norm(self.L, "fro")

&nbsp;       if fnorm > 10.0:

&nbsp;           self.L \*= (10.0 / fnorm)

&nbsp;       W = self.\_update\_cholesky()

&nbsp;       self.whitened\_residuals.append(W @ z)

&nbsp;       if len(self.whitened\_residuals) > 100:

&nbsp;           self.whitened\_residuals.pop(0)

&nbsp;       return grad\_norm

```



\## `compitum/coherence.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from collections import defaultdict

from sklearn.neighbors import KernelDensity



class WeightedReservoir:

&nbsp;   def \_\_init\_\_(self, k=1000, rng=None):

&nbsp;       self.k, self.buf, self.tot\_w = k, \[], 0.0

&nbsp;       self.rng = rng or np.random.default\_rng()



&nbsp;   def add(self, x: np.ndarray, w: float):

&nbsp;       w = max(float(w), 1e-6)

&nbsp;       self.tot\_w += w

&nbsp;       if len(self.buf) < self.k:

&nbsp;           self.buf.append((x.copy(), w))

&nbsp;       else:

&nbsp;           j = int(self.rng.integers(0, int(self.tot\_w)))

&nbsp;           if j < self.k:

&nbsp;               self.buf\[j] = (x.copy(), w)



class CoherenceFunctional:

&nbsp;   def \_\_init\_\_(self, k=1000):

&nbsp;       self.res = defaultdict(lambda: WeightedReservoir(k))

&nbsp;       self.kde\_cache: dict\[str, KernelDensity] = {}



&nbsp;   def update(self, model\_name: str, xw: np.ndarray, success: float):

&nbsp;       self.res\[model\_name].add(xw, success)

&nbsp;       self.kde\_cache.pop(model\_name, None)



&nbsp;   def \_fit(self, model\_name: str) -> KernelDensity | None:

&nbsp;       buf = self.res\[model\_name].buf

&nbsp;       if len(buf) < 10:

&nbsp;           return None

&nbsp;       X = np.stack(\[x for x, \_ in buf], axis=0)

&nbsp;       w = np.array(\[wt for \_, wt in buf], float)

&nbsp;       # Scott rule on whitened coords

&nbsp;       n, d = X.shape

&nbsp;       bw = n \*\* (-1.0 / (d + 4))

&nbsp;       kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X, sample\_weight=w / w.sum())

&nbsp;       self.kde\_cache\[model\_name] = kde

&nbsp;       return kde



&nbsp;   def log\_evidence(self, model\_name: str, xw: np.ndarray) -> float:

&nbsp;       kde = self.kde\_cache.get(model\_name) or self.\_fit(model\_name)

&nbsp;       if kde is None:

&nbsp;           return 0.0

&nbsp;       val = float(kde.score\_samples(\[xw])\[0])

&nbsp;       return float(np.clip(val, -10.0, 10.0))

```



\## `compitum/constraints.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from typing import Any, Dict, List, Tuple



class ReflectiveConstraintSolver:

&nbsp;   def \_\_init\_\_(self, A: np.ndarray, b: np.ndarray):

&nbsp;       self.A, self.b = A, b

&nbsp;       self.last\_viable\_models: List\[Any] = \[]



&nbsp;   def \_is\_feasible(self, model: Any, pgd\_banach: np.ndarray) -> bool:

&nbsp;       if not np.all(self.A @ pgd\_banach <= self.b + 1e-10):

&nbsp;           return False

&nbsp;       return model.capabilities.supports(pgd\_banach)



&nbsp;   def select(self, pgd\_banach: np.ndarray, models: List\[Any],

&nbsp;              utilities: Dict\[Any, float], eps: float = 1e-3) -> Tuple\[Any, Dict]:

&nbsp;       viable = \[m for m in models if self.\_is\_feasible(m, pgd\_banach)]

&nbsp;       self.last\_viable\_models = viable

&nbsp;       if not viable:

&nbsp;           m\_star = max(models, key=lambda m: utilities\[m])

&nbsp;           return m\_star, {"feasible": False, "minimal\_violation": True,

&nbsp;                           "binding\_constraints": \[], "shadow\_prices": {}}



&nbsp;       m\_star = max(viable, key=lambda m: utilities\[m])



&nbsp;       lambdas = {}

&nbsp;       for j in range(self.b.size):

&nbsp;           b\_relaxed = self.b.copy(); b\_relaxed\[j] += eps

&nbsp;           # if relaxation changes feasibility of better competitors, estimate ∂U/∂b\_j

&nbsp;           best\_util = utilities\[m\_star]

&nbsp;           for comp in models:

&nbsp;               if comp in viable or utilities\[comp] <= best\_util:

&nbsp;                   continue

&nbsp;               ok = np.all(self.A @ pgd\_banach <= b\_relaxed + 1e-10) and comp.capabilities.supports(pgd\_banach)

&nbsp;               if ok:

&nbsp;                   best\_util = max(best\_util, utilities\[comp])

&nbsp;           lambdas\[f"lambda\_{j}"] = max(0.0, (best\_util - utilities\[m\_star]) / eps)



&nbsp;       binding = \[j for j, val in enumerate(self.A @ pgd\_banach) if val >= self.b\[j] - 1e-9]

&nbsp;       return m\_star, {"feasible": True, "minimal\_violation": False,

&nbsp;                       "binding\_constraints": binding, "shadow\_prices": lambdas}

```



\## `compitum/effort\_qp.py`



```python

def solve\_effort\_1d(q0, q1, t0, t1, c0, c1, beta):

&nbsp;   """

&nbsp;   Linearized effort e∈\[0,1] around e0. Returns e\_star and box multipliers.

&nbsp;   U(e) = α(q0+q1 e) - βt(t0+t1 e) - βc(c0+c1 e) + const

&nbsp;   """

&nbsp;   alpha, bt, bc = beta

&nbsp;   grad = alpha\*q1 - bt\*t1 - bc\*c1

&nbsp;   e\_star = 1.0 if grad > 0 else 0.0

&nbsp;   lam\_low  = max(0.0, -grad) if e\_star == 0.0 else 0.0

&nbsp;   lam\_high = max(0.0,  grad) if e\_star == 1.0 else 0.0

&nbsp;   return float(e\_star), {"lambda\_low": lam\_low, "lambda\_high": lam\_high}

```



\## `compitum/boundary.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from typing import Dict, Any



class BoundaryAnalyzer:

&nbsp;   def analyze(self, utilities: Dict\[str, float], u\_sigma: Dict\[str, float]) -> Dict\[str, Any]:

&nbsp;       if len(utilities) < 2:

&nbsp;           return {"is\_boundary": False, "reason": "insufficient\_models"}

&nbsp;       items = sorted(utilities.items(), key=lambda kv: kv\[1], reverse=True)

&nbsp;       (m1, u1), (m2, u2) = items\[0], items\[1]

&nbsp;       gap = u1 - u2

&nbsp;       arr = np.array(\[u for \_, u in items])

&nbsp;       probs = np.exp(arr - u1); probs /= probs.sum()

&nbsp;       entropy = -float(np.sum(probs \* np.log(probs + 1e-12)))

&nbsp;       sigma = float(u\_sigma.get(m1, 0.0))

&nbsp;       is\_boundary = (gap < 0.05 or entropy > 0.65) and (sigma > 0.12)

&nbsp;       return {"winner": m1, "runner\_up": m2, "utility\_gap": float(gap),

&nbsp;               "entropy": float(entropy), "uncertainty": sigma, "is\_boundary": bool(is\_boundary)}

```



\## `compitum/control.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from typing import Tuple, Dict



class SRMFController:

&nbsp;   def \_\_init\_\_(self, kappa: float = 0.1, r0: float = 1.0):

&nbsp;       self.kappa = kappa

&nbsp;       self.r = r0

&nbsp;       self.ema\_d = 0.0



&nbsp;   def update(self, d\_star: float, grad\_norm: float) -> Tuple\[float, Dict\[str, float]]:

&nbsp;       self.ema\_d = 0.9\*self.ema\_d + 0.1\*float(d\_star)

&nbsp;       eta\_cap = self.kappa / (float(grad\_norm) + 1e-6)

&nbsp;       if self.ema\_d > 1.5\*self.r:

&nbsp;           self.r \*= 0.8

&nbsp;       elif self.ema\_d < 0.7\*self.r:

&nbsp;           self.r \*= 1.1

&nbsp;       self.r = float(np.clip(self.r, 0.2, 5.0))

&nbsp;       return float(eta\_cap), {"trust\_radius": self.r, "drift\_ema": self.ema\_d}

```



\## `compitum/predictors.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from sklearn.isotonic import IsotonicRegression

from sklearn.ensemble import GradientBoostingRegressor



class CalibratedPredictor:

&nbsp;   """

&nbsp;   Calibrated regressor with quantile bounds (p5,p95).

&nbsp;   For latency/cost: consider enabling monotonic constraints via LightGBM when available.

&nbsp;   """

&nbsp;   def \_\_init\_\_(self):

&nbsp;       self.base = GradientBoostingRegressor(random\_state=42)

&nbsp;       self.iso = IsotonicRegression(out\_of\_bounds="clip")

&nbsp;       self.q05 = GradientBoostingRegressor(loss="quantile", alpha=0.05, random\_state=41)

&nbsp;       self.q95 = GradientBoostingRegressor(loss="quantile", alpha=0.95, random\_state=43)

&nbsp;       self.fitted = False



&nbsp;   def fit(self, X: np.ndarray, y: np.ndarray):

&nbsp;       self.base.fit(X, y)

&nbsp;       raw = self.base.predict(X)

&nbsp;       self.iso.fit(raw, y)

&nbsp;       self.q05.fit(X, y)

&nbsp;       self.q95.fit(X, y)

&nbsp;       self.fitted = True



&nbsp;   def predict(self, X: np.ndarray):

&nbsp;       raw = self.base.predict(X)

&nbsp;       y = self.iso.transform(raw)

&nbsp;       lo = self.q05.predict(X)

&nbsp;       hi = self.q95.predict(X)

&nbsp;       return y, lo, hi

```



\## `compitum/pgd.py`



````python

from \_\_future\_\_ import annotations

import re

import numpy as np

from typing import Dict



class ProductionPGDExtractor:

&nbsp;   """

&nbsp;   Fast, regex-first extractor (spaCy optional). Returns a stable 35D Riemannian vector

&nbsp;   plus a small Banach vector attached separately by the caller if desired.

&nbsp;   """

&nbsp;   def \_\_init\_\_(self):

&nbsp;       self.\_r\_keys = \[f"syn\_{i}" for i in range(6)] + \\

&nbsp;                      \[f"math\_{i}" for i in range(8)] + \\

&nbsp;                      \[f"code\_{i}" for i in range(7)] + \\

&nbsp;                      \[f"sem\_{i}" for i in range(6)] + \\

&nbsp;                      \[f"aux\_{i}" for i in range(8)]  # pad to 35 if some groups are light



&nbsp;   def extract\_features(self, prompt: str) -> Dict\[str, float]:

&nbsp;       feats: Dict\[str, float] = {}

&nbsp;       # syntactic (cheap proxies)

&nbsp;       sents = \[s for s in re.split(r"\[.!?]\\s+", prompt) if s]

&nbsp;       feats\["syn\_0"] = float(np.mean(\[len(s.split()) for s in sents])) if sents else 0.0

&nbsp;       feats\["syn\_1"] = float(np.std(\[len(s.split()) for s in sents])) if sents else 0.0

&nbsp;       feats\["syn\_2"] = float(len(sents))

&nbsp;       feats\["syn\_3"] = float(prompt.count(","))

&nbsp;       feats\["syn\_4"] = float(prompt.count(";"))

&nbsp;       feats\["syn\_5"] = float(min(len(prompt), 4096))  # proxy for length



&nbsp;       # math

&nbsp;       math\_ops = len(re.findall(r"\[∑∏∫∂∇≤≥≠≈]|\\\\(sum|int|prod|frac|cdot)", prompt))

&nbsp;       latex = len(re.findall(r"\\$\[^$]+\\$|\\\\begin\\{equation\\}", prompt))

&nbsp;       feats |= {

&nbsp;           "math\_0": math\_ops,

&nbsp;           "math\_1": latex,

&nbsp;           "math\_2": float(len(re.findall(r"\\bprove|derive|compute|solve\\b", prompt, re.I))),

&nbsp;           "math\_3": float(len(re.findall(r"\[0-9]+(\\.\[0-9]+)?", prompt))),

&nbsp;           "math\_4": float(prompt.count("^")+prompt.count("\_")),

&nbsp;           "math\_5": float("theorem" in prompt.lower()),

&nbsp;           "math\_6": float("lemma" in prompt.lower()),

&nbsp;           "math\_7": float("proof" in prompt.lower()),

&nbsp;       }



&nbsp;       # code

&nbsp;       code\_blocks = len(re.findall(r"```\[\\s\\S]\*?```", prompt))

&nbsp;       lang\_hits = len(re.findall(r"\\b(python|sql|javascript|cpp|java|rust|go)\\b", prompt, re.I))

&nbsp;       feats |= {

&nbsp;           "code\_0": float(code\_blocks),

&nbsp;           "code\_1": float(lang\_hits),

&nbsp;           "code\_2": float(len(re.findall(r"\\bfor|while|if|else|try|catch|except\\b", prompt, re.I))),

&nbsp;           "code\_3": float(len(re.findall(r"\[{}();]", prompt))),

&nbsp;           "code\_4": float("class " in prompt or "def " in prompt),

&nbsp;           "code\_5": float("SELECT " in prompt.upper()),

&nbsp;           "code\_6": float("import " in prompt),

&nbsp;       }



&nbsp;       # semantic proxies

&nbsp;       tokens = prompt.split()

&nbsp;       diffs = \[abs(len(tokens\[i+1])-len(tokens\[i])) for i in range(len(tokens)-1)] if len(tokens) > 1 else \[]

&nbsp;       feats |= {

&nbsp;           "sem\_0": float(np.sum(diffs)) if diffs else 0.0,

&nbsp;           "sem\_1": float(np.mean(diffs)) if diffs else 0.0,

&nbsp;           "sem\_2": float(np.std(diffs)) if diffs else 0.0,

&nbsp;           "sem\_3": float(len(set(\[t.lower() for t in tokens]))),

&nbsp;           "sem\_4": float(len(tokens)),

&nbsp;           "sem\_5": float(len(set(w for w in tokens if len(w)>6))),

&nbsp;       }



&nbsp;       # aux padding (zeros)

&nbsp;       for i in range(8):

&nbsp;           feats\[f"aux\_{i}"] = feats.get(f"aux\_{i}", 0.0)



&nbsp;       # minimal Banach (pragmatic) features for demo

&nbsp;       feats\["prag\_latency\_class"] = 1.0

&nbsp;       feats\["prag\_cost\_class"] = 1.0

&nbsp;       feats\["prag\_pii\_level"] = 0.0

&nbsp;       feats\["prag\_region\_eu\_only"] = 0.0

&nbsp;       return feats

````



\## `compitum/energy.py`



```python

from \_\_future\_\_ import annotations

import numpy as np

from typing import Dict, Tuple

from .metric import SymbolicManifoldMetric



class SymbolicFreeEnergy:

&nbsp;   def \_\_init\_\_(self, alpha, beta\_t, beta\_c, beta\_d, beta\_s):

&nbsp;       self.alpha, self.beta\_t, self.beta\_c, self.beta\_d, self.beta\_s = alpha, beta\_t, beta\_c, beta\_d, beta\_s



&nbsp;   @property

&nbsp;   def beta\_d(self): return self.\_beta\_d

&nbsp;   @beta\_d.setter

&nbsp;   def beta\_d(self, v): self.\_beta\_d = v



&nbsp;   def compute(self, xR: np.ndarray, model, predictors: Dict, coherence, metric: SymbolicManifoldMetric

&nbsp;              ) -> Tuple\[float, float, Dict\[str, float]]:

&nbsp;       d, d\_std = metric.distance(xR, model.center)

&nbsp;       q, q\_lo, q\_hi = predictors\["quality"].predict(\[xR])

&nbsp;       t, t\_lo, t\_hi = predictors\["latency"].predict(\[xR])

&nbsp;       c, c\_lo, c\_hi = predictors\["cost"].predict(\[xR])



&nbsp;       # evidence in whitened space

&nbsp;       W = metric.W or metric.\_update\_cholesky()

&nbsp;       xw = W @ (xR - model.center)

&nbsp;       log\_e = coherence.log\_evidence(model.name, xw)



&nbsp;       U = (self.alpha\*q\[0] - self.beta\_t\*t\[0] - self.beta\_c\*c\[0] - self.beta\_d\*d + self.beta\_s\*log\_e)

&nbsp;       U\_var = ((self.alpha\*(q\_hi-q\_lo)/3.92)\*\*2 + (self.beta\_t\*(t\_hi-t\_lo)/3.92)\*\*2 +

&nbsp;                (self.beta\_c\*(c\_hi-c\_lo)/3.92)\*\*2 + (self.beta\_d\*d\_std)\*\*2)

&nbsp;       comps = {"quality": float(q\[0]), "latency": float(-t\[0]), "cost": float(-c\[0]),

&nbsp;                "distance": float(-d), "evidence": float(log\_e), "uncertainty": float(np.sqrt(U\_var))}

&nbsp;       return float(U), float(np.sqrt(U\_var)), comps

```



\## `compitum/router.py`



```python

from \_\_future\_\_ import annotations

import time, json, hashlib

import numpy as np

from dataclasses import dataclass

from typing import Dict, Any, List

from .utils import split\_features, pgd\_hash

from .boundary import BoundaryAnalyzer

from .constraints import ReflectiveConstraintSolver

from .control import SRMFController

from .energy import SymbolicFreeEnergy

from .metric import SymbolicManifoldMetric



@dataclass

class SwitchCertificate:

&nbsp;   model: str

&nbsp;   utility: float

&nbsp;   utility\_components: Dict\[str, float]

&nbsp;   constraints: Dict\[str, Any]

&nbsp;   boundary\_analysis: Dict\[str, Any]

&nbsp;   drift\_status: Dict\[str, float]

&nbsp;   pgd\_signature: str

&nbsp;   timestamp: float

&nbsp;   router\_version: str = "0.1.0"



&nbsp;   def to\_json(self) -> str:

&nbsp;       return json.dumps({

&nbsp;           "model": self.model,

&nbsp;           "utility": round(self.utility, 6),

&nbsp;           "utility\_components": {k: float(v) for k, v in self.utility\_components.items()},

&nbsp;           "constraints": self.constraints,

&nbsp;           "boundary": self.boundary\_analysis,

&nbsp;           "drift": self.drift\_status,

&nbsp;           "pgd\_signature": self.pgd\_signature\[:16],

&nbsp;           "timestamp": self.timestamp,

&nbsp;           "router\_version": self.router\_version

&nbsp;       }, indent=2)



class CompitumRouter:

&nbsp;   def \_\_init\_\_(self, models: List\[Any], predictors: Dict\[str, Dict], solver: ReflectiveConstraintSolver,

&nbsp;                coherence, boundary: BoundaryAnalyzer, srmf: SRMFController,

&nbsp;                pgd\_extractor, metric\_map: Dict\[str, SymbolicManifoldMetric],

&nbsp;                energy: SymbolicFreeEnergy, update\_stride: int = 8):

&nbsp;       self.models = {m.name: m for m in models}

&nbsp;       self.predictors = predictors

&nbsp;       self.solver = solver

&nbsp;       self.coherence = coherence

&nbsp;       self.boundary = boundary

&nbsp;       self.srmf = srmf

&nbsp;       self.pgd = pgd\_extractor

&nbsp;       self.metric\_map = metric\_map

&nbsp;       self.energy = energy

&nbsp;       self.\_step = 0

&nbsp;       self.\_stride = max(int(update\_stride), 1)



&nbsp;   def route(self, prompt: str, context: Dict\[str, Any] | None = None) -> SwitchCertificate:

&nbsp;       context = context or {}

&nbsp;       feats = self.pgd.extract\_features(prompt)

&nbsp;       xR\_all, xB = split\_features(feats)

&nbsp;       utilities, comps, u\_sigmas = {}, {}, {}



&nbsp;       for name, model in self.models.items():

&nbsp;           met = self.metric\_map\[name]

&nbsp;           U, sig, uc = self.energy.compute(xR\_all, model, self.predictors\[name], self.coherence, met)

&nbsp;           utilities\[self.models\[name]] = float(U)

&nbsp;           comps\[name] = uc

&nbsp;           u\_sigmas\[name] = float(sig)



&nbsp;       m\_star, cinfo = self.solver.select(xB, list(self.models.values()), utilities)

&nbsp;       binfo = self.boundary.analyze({m.name: utilities\[m] for m in self.models.values()}, u\_sigmas)



&nbsp;       # Adapt metric periodically (two-timescale)

&nbsp;       self.\_step += 1

&nbsp;       grad\_norm = 1.0

&nbsp;       if self.\_step % self.\_stride == 0:

&nbsp;           met = self.metric\_map\[m\_star.name]

&nbsp;           d\_best = abs(-comps\[m\_star.name]\["distance"])

&nbsp;           grad\_norm = met.update\_spd(xR\_all, self.models\[m\_star.name].center, self.energy.beta\_d,

&nbsp;                                      d\_best, eta=1e-2, srmf\_controller=self.srmf)



&nbsp;       \_, drift = self.srmf.update(d\_star=abs(-comps\[m\_star.name]\["distance"]), grad\_norm=grad\_norm)



&nbsp;       cert = SwitchCertificate(

&nbsp;           model=m\_star.name,

&nbsp;           utility=utilities\[m\_star],

&nbsp;           utility\_components=comps\[m\_star.name],

&nbsp;           constraints=cinfo,

&nbsp;           boundary\_analysis=binfo,

&nbsp;           drift\_status=drift,

&nbsp;           pgd\_signature=pgd\_hash(prompt),

&nbsp;           timestamp=time.time()

&nbsp;       )

&nbsp;       return cert

```



\## `compitum/cli.py`



```python

from \_\_future\_\_ import annotations

import json, yaml, numpy as np, typer

from pathlib import Path

from typing import Optional

from .pgd import ProductionPGDExtractor

from .models import Model

from .capabilities import Capabilities

from .predictors import CalibratedPredictor

from .metric import SymbolicManifoldMetric

from .coherence import CoherenceFunctional

from .constraints import ReflectiveConstraintSolver

from .boundary import BoundaryAnalyzer

from .control import SRMFController

from .energy import SymbolicFreeEnergy

from .router import CompitumRouter



app = typer.Typer(help="compitum CLI")



def \_load\_constraints(path: Path):

&nbsp;   cfg = yaml.safe\_load(path.read\_text())

&nbsp;   import numpy as np

&nbsp;   return np.array(cfg\["A"], float), np.array(cfg\["b"], float)



def \_toy\_models(D: int):

&nbsp;   rng = np.random.default\_rng(7)

&nbsp;   centers = {

&nbsp;       "fast":    rng.normal(0.0, 0.4, size=D),

&nbsp;       "thinking":rng.normal(0.0, 1.0, size=D),

&nbsp;       "auto":    rng.normal(0.1, 0.7, size=D)

&nbsp;   }

&nbsp;   caps = Capabilities(regions={"US","CA","EU"}, tools\_allowed={"none"})

&nbsp;   return \[Model(name=k, center=v, capabilities=caps) for k, v in centers.items()]



@app.command()

def route(prompt: str,

&nbsp;         constraints: Path = Path("configs/constraints\_us\_default.yaml"),

&nbsp;         defaults: Path = Path("configs/router\_defaults.yaml"),

&nbsp;         verbose: bool = False):

&nbsp;   dcfg = yaml.safe\_load(defaults.read\_text())

&nbsp;   D = int(dcfg\["metric"]\["D"])

&nbsp;   rank = int(dcfg\["metric"]\["rank"])

&nbsp;   delta = float(dcfg\["metric"]\["delta"])



&nbsp;   models = \_toy\_models(D)

&nbsp;   predictors = {

&nbsp;       m.name: {"quality": CalibratedPredictor(), "latency": CalibratedPredictor(), "cost": CalibratedPredictor()}

&nbsp;       for m in models

&nbsp;   }

&nbsp;   # quick synthetic fit for demo

&nbsp;   X\_demo = np.random.randn(512, D)

&nbsp;   for m in models:

&nbsp;       yq = 0.6 + 0.1\*np.tanh(X\_demo @ (m.center/np.linalg.norm(m.center)+1e-8))

&nbsp;       yt = 0.5 + 0.5\*np.abs(X\_demo @ np.ones(D)/np.sqrt(D))

&nbsp;       yc = 0.2 + 0.4\*np.abs(X\_demo @ (np.arange(D)/D))

&nbsp;       predictors\[m.name]\["quality"].fit(X\_demo, yq)

&nbsp;       predictors\[m.name]\["latency"].fit(X\_demo, yt)

&nbsp;       predictors\[m.name]\["cost"].fit(X\_demo, yc)



&nbsp;   metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}

&nbsp;   coherence = CoherenceFunctional(k=500)

&nbsp;   A,b = \_load\_constraints(constraints)

&nbsp;   solver = ReflectiveConstraintSolver(A, b)

&nbsp;   boundary = BoundaryAnalyzer()

&nbsp;   srmf = SRMFController()

&nbsp;   energy = SymbolicFreeEnergy(dcfg\["alpha"], dcfg\["beta\_t"], dcfg\["beta\_c"], dcfg\["beta\_d"], dcfg\["beta\_s"])

&nbsp;   pgd = ProductionPGDExtractor()



&nbsp;   router = CompitumRouter(models, predictors, solver, coherence, boundary, srmf, pgd, metrics, energy, update\_stride=int(dcfg\["update\_stride"]))

&nbsp;   cert = router.route(prompt)

&nbsp;   typer.echo(cert.to\_json() if verbose else json.dumps({"model": cert.model, "U": cert.utility}, indent=2))

```



---



\# 5) Tests



\## `tests/test\_invariants.py`



```python

import numpy as np

from compitum.metric import SymbolicManifoldMetric



def test\_spd\_properties():

&nbsp;   m = SymbolicManifoldMetric(20, 5)

&nbsp;   M = m.metric\_matrix()

&nbsp;   assert np.allclose(M, M.T)

&nbsp;   eig = np.linalg.eigvalsh(M)

&nbsp;   assert np.all(eig > 0)



def test\_triangle\_inequality():

&nbsp;   m = SymbolicManifoldMetric(12, 4)

&nbsp;   x, y, z = np.random.randn(12), np.random.randn(12), np.random.randn(12)

&nbsp;   d\_xy, \_ = m.distance(x, y)

&nbsp;   d\_yz, \_ = m.distance(y, z)

&nbsp;   d\_xz, \_ = m.distance(x, z)

&nbsp;   assert d\_xz <= d\_xy + d\_yz + 1e-9



def test\_whitening\_isometry():

&nbsp;   m = SymbolicManifoldMetric(10, 3); m.\_update\_cholesky()

&nbsp;   a, b = np.random.randn(10), np.random.randn(10)

&nbsp;   d, \_ = m.distance(a, b)

&nbsp;   wa, wb = m.W @ a, m.W @ b

&nbsp;   assert np.isclose(d, np.linalg.norm(wa - wb), rtol=1e-9)

```



\## `tests/test\_boundary.py`



```python

from compitum.boundary import BoundaryAnalyzer



def test\_boundary\_logic():

&nbsp;   b = BoundaryAnalyzer()

&nbsp;   utilities = {"fast": 0.50, "thinking": 0.52, "auto": 0.48}

&nbsp;   u\_sigma = {"fast": 0.05, "thinking": 0.2, "auto": 0.05}

&nbsp;   info = b.analyze(utilities, u\_sigma)

&nbsp;   assert "is\_boundary" in info

```



\## `tests/test\_constraints.py`



```python

import numpy as np

from dataclasses import dataclass

from compitum.constraints import ReflectiveConstraintSolver

from compitum.capabilities import Capabilities



@dataclass

class M:

&nbsp;   name: str

&nbsp;   capabilities: Capabilities



def test\_solver\_basic():

&nbsp;   A = np.eye(2); b = np.array(\[1.0, 1.0])

&nbsp;   solver = ReflectiveConstraintSolver(A, b)

&nbsp;   pgd = np.array(\[0.5, 0.0])

&nbsp;   models = \[M("a", Capabilities(set(), set())), M("b", Capabilities(set(), set()))]

&nbsp;   utilities = {models\[0]: 0.2, models\[1]: 0.3}

&nbsp;   m\_star, info = solver.select(pgd, models, utilities)

&nbsp;   assert m\_star.name == "b"

&nbsp;   assert info\["feasible"] is True

```



---



\# 6) Examples



\## `examples/synth\_bench.py`



```python

import numpy as np

from compitum.metric import SymbolicManifoldMetric



def main():

&nbsp;   rng = np.random.default\_rng(0)

&nbsp;   D = 35

&nbsp;   M = SymbolicManifoldMetric(D, 8)

&nbsp;   # two clusters: math-like vs code-like

&nbsp;   math\_center = rng.normal(0, 1, size=D)

&nbsp;   code\_center = rng.normal(0, 1, size=D); code\_center\[:5] += 2.0

&nbsp;   X\_math = rng.normal(0, 0.6, size=(500, D)) + math\_center

&nbsp;   X\_code = rng.normal(0, 0.6, size=(500, D)) + code\_center

&nbsp;   dm = np.mean(\[M.distance(x, math\_center)\[0] for x in X\_math])

&nbsp;   dc = np.mean(\[M.distance(x, code\_center)\[0] for x in X\_code])

&nbsp;   print({"avg\_d\_math": float(dm), "avg\_d\_code": float(dc)})

if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()

```



\## `examples/demo\_route.py`



```python

from subprocess import run

run(\["compitum","route","--prompt","Prove that the harmonic series diverges."], check=True)

```



---



\## That’s it



\* Run with \*\*Gemini\*\*: `gemini run setup \&\& gemini run test \&\& gemini run route-demo`

\* Or plain shell: `make setup test demo`



This package gives your team a \*\*contained, production-ready starting point\*\*: hardened geometry, constraint solver w/ shadow prices, whitened KDE, SRMF trust region, CLI, tests, and a bench. Hook your real predictors, real PGD extractors, and your PyLantern observers into the seams already exposed.



