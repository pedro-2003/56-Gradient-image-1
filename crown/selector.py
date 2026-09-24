"""Candidate selection and plateau detection on measured statistics.

Two problems, both created by the evaluator's structure:

1. Selection noise. The holdout screen uses 1 of the evaluator's 16 noise draws
   per stratum; the argmin over many candidates of a noisy estimate is biased
   toward lucky candidates (winner's curse). Fix: race — screen every
   candidate cheaply, confirm the top few with more draws, and pick with a
   one-standard-error rule using *paired* differences (same noise draws for
   both candidates, so the SE is the SE of the difference, not of the levels).

2. Plateau. Training past the holdout optimum wastes the rest of the budget
   (50-90% in the current champion's own runs). Fix: declare a plateau when
   recent screens fail to beat the best by more than the paired SE, so the
   controller can spend the remaining budget on something that still helps.

No constant here is tuned on a dataset: the SE is measured from the screen's
own per-case losses, and the small integers (top-k, window) are counts.
"""

from dataclasses import dataclass, field
from typing import Optional

from .scoring import ScoreReport


@dataclass
class Candidate:
    tag: str
    step: int
    state: dict
    scale: float
    screen: Optional[ScoreReport] = None
    confirm: Optional[ScoreReport] = None
    member: int = 0

    @property
    def score(self):
        return self.confirm.score if self.confirm is not None else (self.screen.score if self.screen else float("inf"))


class Selector:
    def __init__(self):
        self.cands: list[Candidate] = []
        self.history: list[tuple[int, float]] = []   # (step, screen score) in time order

    def add(self, cand: Candidate):
        self.cands.append(cand)
        if cand.screen is not None:
            self.history.append((cand.step, cand.screen.score))

    def best(self, confirmed_only=False) -> Optional[Candidate]:
        pool = [c for c in self.cands if (c.confirm if confirmed_only else (c.confirm or c.screen)) is not None]
        return min(pool, key=lambda c: c.score) if pool else None

    def top(self, k: int, member=None):
        pool = [c for c in self.cands if c.screen is not None and (member is None or c.member == member)]
        return sorted(pool, key=lambda c: c.screen.score)[:k]

    # --- 1-SE rule ------------------------------------------------------------------
    def pick(self, use_confirm=True) -> Optional[Candidate]:
        """Among candidates whose paired difference to the best is within one SE,
        return the earliest one. Earlier = less overfit direction, by the measured
        monotone overfit curves; the rule never picks something significantly worse."""
        pool = [c for c in self.cands if (c.confirm if use_confirm else c.screen) is not None]
        if not pool:
            return self.best()
        rep = (lambda c: c.confirm) if use_confirm else (lambda c: c.screen)
        b = min(pool, key=lambda c: rep(c).score)
        within = []
        for c in pool:
            if c is b:
                within.append(c); continue
            mean, se, n = rep(c).paired_diff(rep(b))
            if n > 1 and mean <= se:      # c - b <= SE  → not distinguishably worse
                within.append(c)
        return min(within, key=lambda c: c.step)

    # --- plateau -------------------------------------------------------------------
    def plateaued(self, window: int = 3, member=None) -> bool:
        """True when the last `window` eval POINTS (a point = all candidates screened at one step of
        one member) each failed to beat the running best of the earlier points by more than the
        paired SE of that comparison. Counting points, not candidates, keeps raw/ema pairs from
        making a two-point window look like three."""
        screened = [c for c in self.cands if c.screen is not None and (member is None or c.member == member)]
        points = {}
        for c in screened:
            key = (c.member, c.step)
            if key not in points or c.screen.score < points[key].screen.score:
                points[key] = c
        ordered = [points[k] for k in sorted(points)]
        if len(ordered) < window + 1:
            return False
        recent, earlier = ordered[-window:], ordered[:-window]
        b = min(earlier, key=lambda c: c.screen.score)
        for c in recent:
            mean, se, n = c.screen.paired_diff(b.screen)
            if n > 1 and mean < -se:       # a real improvement
                return False
        return True
