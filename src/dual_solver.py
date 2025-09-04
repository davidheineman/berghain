from typing import List, Dict, Optional
import math
from api import Constraint
from solvers import BaseSolver

class DualThresholdSolver(BaseSolver):
    SCENARIO_DEFAULTS: Dict[int, Dict[str, object]] = {
        # Tuned from sweeps
        2: {
            "z0": 2.5,
            "z1": 0.5,
            "eta0": 0.55,
            "lambda_max": 12.5,
            "endgame_R": 32,
            "use_rarity": False,
        },
        3: {
            "z0": 2.5,
            "z1": 0.5,
            "eta0": 0.77,
            "lambda_max": 12.0,
            "endgame_R": 33,
            "use_rarity": False,
        },
    }
    def __init__(
        self,
        attribute_frequencies: Dict[str, float],
        correlation_matrix: Dict[str, Dict[str, float]],
        constraints: List[Constraint],
        distribution: List,
        api_client=None,
        N: int = 1000,
        z0: float = 2.5,
        z1: float = 0.5,
        eta0: float = 0.77,
        lambda_max: float = 12.0,
        use_rarity: bool = False,
        endgame_R: int = 33,
        # Scenario-aware defaults
        scenario: Optional[int] = None,
        use_scenario_defaults: bool = True,
    ):
        super().__init__(api_client)
        self.N = int(N)
        # Start with provided values
        self.z0 = float(z0)
        self.z1 = float(z1)
        self.eta0 = float(eta0)
        self.lambda_max = float(lambda_max)
        self.use_rarity = bool(use_rarity)
        self.endgame_R = int(endgame_R)
        # Apply scenario-specific defaults when requested
        if use_scenario_defaults and scenario in self.SCENARIO_DEFAULTS:
            defaults = self.SCENARIO_DEFAULTS[int(scenario)]
            self.z0 = float(defaults.get("z0", self.z0))
            self.z1 = float(defaults.get("z1", self.z1))
            self.eta0 = float(defaults.get("eta0", self.eta0))
            self.lambda_max = float(defaults.get("lambda_max", self.lambda_max))
            self.endgame_R = int(defaults.get("endgame_R", self.endgame_R))
            self.use_rarity = bool(defaults.get("use_rarity", self.use_rarity))
        self.rho: Dict[str, float] = {}
        self.m: Dict[str, int] = {}
        self.p: Dict[str, float] = {}
        self.rarity: Dict[str, float] = {}
        self.lam: Dict[str, float] = {}
        self.tracked_attributes: List[str] = []
        self.t: int = 0
        self.is_initialized: bool = False
        # Accept optional stats for symmetry with other solvers
        self.attribute_frequencies: Dict[str, float] = attribute_frequencies
        self.correlation_matrix: Dict[str, Dict[str, float]] = correlation_matrix
        self.initialize_policy(constraints)

    def update_statistics(self, attribute_frequencies: Dict[str, float], correlation_matrix: Dict[str, Dict[str, float]]):
        self.attribute_frequencies = dict(attribute_frequencies)
        self.correlation_matrix = dict(correlation_matrix)

    def initialize_policy(self, constraints: List[Constraint]):
        self.constraints = list(constraints)
        attrs = set(a.attribute for a in constraints)
        attrs.update(self.attribute_frequencies.keys())
        self.tracked_attributes = sorted(attrs)
        if not self.tracked_attributes:
            return False
        self.rho = {c.attribute: max(0.0, min(1.0, c.min_count / float(self.N))) for c in constraints}
        self.m = {a: int(math.ceil(self.N * self.rho.get(a, 0.0))) for a in self.tracked_attributes}
        self.p = {a: float(self.attribute_frequencies.get(a, 0.5)) for a in self.tracked_attributes}
        self.rarity = {a: ((1.0 - self.p[a]) / max(self.p[a], 1e-6)) for a in self.tracked_attributes}
        self.lam = {}
        for a in self.tracked_attributes:
            rho_a = self.rho.get(a, 0.0)
            p_a = min(max(self.p[a], 1e-6), 1 - 1e-6)
            v = math.log(rho_a / max(1 - rho_a, 1e-6)) - math.log(p_a / (1 - p_a)) if 0 < rho_a < 1 else (0.0 if rho_a == 0 else self.lambda_max)
            self.lam[a] = float(max(0.0, min(self.lambda_max, v)))
        self.t = 0
        self.is_initialized = True
        return True

    def _z(self, R: int) -> float:
        return float(self.z1 + (self.z0 - self.z1) * max(R, 0) / float(self.N))

    def _safety_b(self, a: str, z: float) -> int:
        rho_a = self.rho.get(a, 0.0)
        return int(math.ceil(z * math.sqrt(self.N * rho_a * max(1 - rho_a, 0.0))))

    def _score(self, x: Dict[str, bool], u: Dict[str, float]) -> float:
        s = 1.0
        for a in self.tracked_attributes:
            w = self.rarity[a] if self.use_rarity else 1.0
            xa = 1.0 if x.get(a, False) else 0.0
            s += self.lam[a] * w * (xa - u.get(a, 0.0))
        return float(s)

    def _update_lam(self, counts: Dict[str, int], admitted: int):
        self.t += 1
        n = admitted
        if n <= 0:
            return
        eta = self.eta0 / math.sqrt(self.t)
        for a in self.tracked_attributes:
            share = counts.get(a, 0) / float(n)
            delta = self.rho.get(a, 0.0) - share
            self.lam[a] = float(max(0.0, min(self.lambda_max, self.lam[a] + eta * delta)))

    def should_accept(self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int) -> bool:
        if not self.is_initialized:
            raise ValueError("Call update_statistics() and initialize_policy() first.")
        if admitted >= self.N:
            return False
        R = self.N - admitted
        must = []
        for c in self.constraints:
            a = c.attribute
            n_a = int(current_counts.get(a, 0))
            if n_a + max(R - 1, 0) < self.m[a]:
                must.append(a)
        if must:
            ok = all(bool(attributes.get(a, False)) for a in must)
            if ok:
                n_prime = admitted + 1
                counts_prime = dict(current_counts)
                for a in self.tracked_attributes:
                    if attributes.get(a, False):
                        counts_prime[a] = counts_prime.get(a, 0) + 1
                self._update_lam(counts_prime, n_prime)
                return True
            return False
        z = self._z(R)
        d = {}
        u = {}
        for a in self.tracked_attributes:
            n_a = int(current_counts.get(a, 0))
            b_a = self._safety_b(a, z)
            d[a] = max(0, self.m[a] + b_a - n_a)
            u[a] = d[a] / float(max(R, 1))
        if R <= self.endgame_R:
            need = [a for a in self.tracked_attributes if d[a] > 0]
            ok = any(attributes.get(a, False) for a in need) if need else True
            if ok:
                n_prime = admitted + 1
                counts_prime = dict(current_counts)
                for a in self.tracked_attributes:
                    if attributes.get(a, False):
                        counts_prime[a] = counts_prime.get(a, 0) + 1
                self._update_lam(counts_prime, n_prime)
                return True
            return False
        s = self._score(attributes, u)
        accept = s > 0.0
        if accept:
            n_prime = admitted + 1
            counts_prime = dict(current_counts)
            for a in self.tracked_attributes:
                if attributes.get(a, False):
                    counts_prime[a] = counts_prime.get(a, 0) + 1
            self._update_lam(counts_prime, n_prime)
        return bool(accept)

    def get_decision_confidence(self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int) -> float:
        if not self.is_initialized:
            return 0.5
        R = self.N - admitted
        z = self._z(R)
        d = {}
        u = {}
        for a in self.tracked_attributes:
            n_a = int(current_counts.get(a, 0))
            b_a = self._safety_b(a, z)
            d[a] = max(0, self.m[a] + b_a - n_a)
            u[a] = d[a] / float(max(R, 1))
        s = self._score(attributes, u)
        if R <= self.endgame_R and any(d[a] > 0 for a in self.tracked_attributes):
            return 0.9 if any(attributes.get(a, False) for a in self.tracked_attributes if d[a] > 0) else 0.1
        k = 2.0
        return float(1.0 / (1.0 + math.exp(-k * s)))
