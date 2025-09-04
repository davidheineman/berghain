from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import math

@dataclass
class Constraint:
    attribute: str
    min_count: int

class DistributionAwareSolver:
    def __init__(self, N: int = 1000, rng_seed: int = 17, endgame_R: int = 50, z: float = 2.0, lambda_max: float = 64.0):
        self.N = int(N)
        self.endgame_R = int(endgame_R)
        self.z = float(z)
        self.lambda_max = float(lambda_max)
        self.rng = np.random.default_rng(rng_seed)
        self.attr_order: List[str] = []
        self.rho: Dict[str, float] = {}
        self.m: Dict[str, int] = {}
        self.xs: np.ndarray = np.zeros((0,0), dtype=np.uint8)
        self.ps: np.ndarray = np.zeros(0, dtype=float)
        self.a_star: np.ndarray = np.zeros(0, dtype=float)
        self.type_keys: List[Tuple[int,...]] = []
        self.type_index: Dict[Tuple[int,...], int] = {}
        self.ready = False

    def initialize_policy(self, constraints: List[Constraint], distribution: List[Tuple[Dict[str,bool], float]]):
        attrs = [c.attribute for c in constraints]
        self.attr_order = sorted(set(attrs))
        self.rho = {c.attribute: c.min_count / float(self.N) for c in constraints}
        self.m = {a: int(math.ceil(self.N * self.rho[a])) for a in self.attr_order}
        M = len(distribution)
        d = len(self.attr_order)
        xs = np.zeros((M, d), dtype=np.uint8)
        ps = np.zeros(M, dtype=float)
        for i, (feat, p) in enumerate(distribution):
            for j, a in enumerate(self.attr_order):
                xs[i, j] = 1 if feat.get(a, False) else 0
            ps[i] = float(p)
        ps = ps / ps.sum()
        self.xs = xs
        self.ps = ps
        A = xs - np.asarray([self.rho[a] for a in self.attr_order], dtype=float)[None, :]
        lam = np.zeros(d, dtype=float)
        T = 5000
        alpha0 = 1.0
        for t in range(1, T + 1):
            s = 1.0 + A @ lam
            z = (s > 0.0).astype(float)
            g = (ps[:, None] * (z[:, None] * A)).sum(axis=0)
            lam = np.minimum(self.lambda_max, np.maximum(0.0, lam - (alpha0 / math.sqrt(t)) * g))
        s = 1.0 + A @ lam
        a = (s > 0.0).astype(float)
        for _ in range(300):
            sl = (self.ps[:, None] * (a[:, None] * A)).sum(axis=0)
            if np.all(sl >= -1e-10):
                break
            lam = np.minimum(self.lambda_max, np.maximum(0.0, lam + 0.5 * (-np.minimum(sl, 0.0))))
            s = 1.0 + A @ lam
            a = (s > 0.0).astype(float)
        self.a_star = a.astype(float)
        self.type_keys = [tuple(row.tolist()) for row in xs]
        self.type_index = {k: i for i, k in enumerate(self.type_keys)}
        self.ready = True
        return True

    def _safety_b(self, a: str) -> int:
        rho_a = self.rho[a]
        return int(math.ceil(self.z * math.sqrt(self.N * rho_a * max(1 - rho_a, 0.0))))

    def _guard_must_attrs(self, current_counts: Dict[str, int], admitted: int) -> List[str]:
        R = self.N - admitted
        must = []
        for a in self.attr_order:
            n_a = int(current_counts.get(a, 0))
            if n_a + max(R - 1, 0) < self.m[a]:
                must.append(a)
        return must

    def _type_accept_prob(self, attributes: Dict[str, bool]) -> float:
        key = tuple(1 if attributes.get(a, False) else 0 for a in self.attr_order)
        i = self.type_index.get(key, None)
        if i is None:
            diffs = np.sum(np.abs(self.xs - np.asarray(key, dtype=np.uint8)[None, :]), axis=1)
            i = int(np.argmin(diffs))
        return float(self.a_star[i])

    def should_accept(self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int) -> bool:
        if not self.ready or admitted >= self.N:
            return False
        R = self.N - admitted
        must = self._guard_must_attrs(current_counts, admitted)
        if must:
            if all(bool(attributes.get(a, False)) for a in must):
                return True
            return False
        d = {}
        for a in self.attr_order:
            n_a = int(current_counts.get(a, 0))
            b_a = self._safety_b(a)
            d[a] = max(0, self.m[a] + b_a - n_a)
        if R <= self.endgame_R:
            need = [a for a in self.attr_order if d[a] > 0]
            if need and not any(attributes.get(a, False) for a in need):
                return False
        p_acc = self._type_accept_prob(attributes)
        return bool(self.rng.random() < p_acc)
