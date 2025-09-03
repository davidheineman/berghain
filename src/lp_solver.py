import numpy as np
from scipy.optimize import linprog, minimize
from itertools import product
from typing import Dict, List
from api import Constraint
from solvers import BaseSolver


def fit_maxent(marginals, correlations, max_iter=500, lr=0.05):
    """
    Fit a maximum-entropy pairwise Ising model to match given marginals and correlations.

    Args:
        marginals: Array of marginal probabilities for each attribute
        correlations: Matrix of pairwise correlations between attributes
        max_iter: Maximum number of optimization iterations
        lr: Learning rate (not used in L-BFGS-B but kept for compatibility)

    Returns:
        Tuple of (all_possible_combinations, fitted_probabilities)
    """
    d = len(marginals)
    h = np.zeros(d)
    J = np.zeros((d, d))

    def moment_error(params):
        h = params[:d]
        J_flat = params[d:]
        J = np.zeros((d, d))
        idx = np.triu_indices(d, 1)
        J[idx] = J_flat
        J = J + J.T
        probs = []
        for x in product([0, 1], repeat=d):
            x = np.array(x)
            e = h @ x + 0.5 * x @ J @ x
            probs.append(np.exp(e))
        probs = np.array(probs)
        probs /= probs.sum()
        xs = np.array(list(product([0, 1], repeat=d)))
        means = probs @ xs
        pair = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                pair[i, j] = (probs @ (xs[:, i] * xs[:, j])) - means[i] * means[j]
        pair = pair + pair.T
        err = np.sum((means - marginals) ** 2)
        err += np.sum(
            (pair[np.triu_indices(d, 1)] - correlations[np.triu_indices(d, 1)]) ** 2
        )
        return err

    params0 = np.zeros(d + d * (d - 1) // 2)
    res = minimize(
        moment_error, params0, method="L-BFGS-B", options={"maxiter": max_iter}
    )
    h = res.x[:d]
    J_flat = res.x[d:]
    J = np.zeros((d, d))
    idx = np.triu_indices(d, 1)
    J[idx] = J_flat
    J = J + J.T
    xs = np.array(list(product([0, 1], repeat=d)))
    probs = np.array([np.exp(h @ x + 0.5 * x @ J @ x) for x in xs])
    probs /= probs.sum()
    return xs, probs


def solve_lp(xs, probs, groups, r):
    """
    Solve the linear program for optimal acceptance probabilities.

    Args:
        xs: All possible attribute combinations
        probs: Probabilities of each combination
        groups: List of attribute groups for constraints
        r: List of minimum required fractions for each group

    Returns:
        Array of acceptance probabilities for each combination
    """
    T = len(xs)

    # Objective: maximize expected admissions
    c = -probs

    # Constraints: ensure we meet minimum requirements for each attribute
    A = []
    b = []

    for G, rk in zip(groups, r):
        mask = np.array([int(all(x[i] == 1 for i in G)) for x in xs])
        row = -mask * probs
        A.append(row)
        b.append(-rk)

    # Capacity constraint
    A.append(probs)
    b.append(1.0)

    # Solve the linear program
    res = linprog(
        c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, 1)] * T, method="highs"
    )

    if res.success:
        return res.x
    else:
        # Simple fallback: uniform acceptance
        return np.ones(T) * 0.5


class OnlinePolicy:
    """Online policy for making acceptance decisions based on pre-computed probabilities."""

    def __init__(self, xs, a):
        self.xs = xs
        self.a = a

    def accept(self, person):
        """Determine whether to accept a person based on their attributes."""
        for i, x in enumerate(self.xs):
            if np.all(x == person):
                return np.random.rand() < self.a[i]
        return False


class LinearProgrammingSolver(BaseSolver):
    """
    Linear Programming-based solver that optimizes acceptance decisions.

    This solver uses a maximum entropy model to estimate the distribution of future people
    and solves a linear program to find optimal acceptance probabilities that satisfy
    all constraints with high probability.
    """

    def __init__(self, api_client=None):
        """
        Initialize the LP solver.

        Args:
            api_client: API client for the game
        """
        super().__init__(api_client)
        self.attribute_frequencies = {}
        self.correlation_matrix = {}
        self.attribute_order = []
        self.policy = None
        self.is_initialized = False

    def update_statistics(self, game_data: Dict):
        """Update attribute statistics from game data."""
        if "attributeStatistics" in game_data:
            stats = game_data["attributeStatistics"]
            if "relativeFrequencies" in stats:
                self.attribute_frequencies = stats["relativeFrequencies"]
            if "correlations" in stats:
                self.correlation_matrix = stats["correlations"]

    def initialize_policy(self, constraints: List[Constraint]):
        """
        Initialize the LP policy based on current statistics and constraints.
        This should be called once we have enough data about attribute frequencies.
        """
        if not self.attribute_frequencies or not self.correlation_matrix:
            return False

        # Get all unique attributes from constraints
        all_attributes = set()
        for constraint in constraints:
            all_attributes.add(constraint.attribute)

        # Add any attributes from statistics that aren't in constraints
        all_attributes.update(self.attribute_frequencies.keys())

        # Create ordered list of attributes
        self.attribute_order = sorted(list(all_attributes))
        d = len(self.attribute_order)

        if d == 0:
            return False

        # Extract marginals and correlations in the correct order
        marginals = np.zeros(d)
        correlations = np.zeros((d, d))

        for i, attr in enumerate(self.attribute_order):
            if attr in self.attribute_frequencies:
                marginals[i] = self.attribute_frequencies[attr]
            else:
                marginals[i] = 0.5  # Default if not in statistics

        for i, attr1 in enumerate(self.attribute_order):
            for j, attr2 in enumerate(self.attribute_order):
                if i != j:
                    key = f"{attr1}_{attr2}"
                    if key in self.correlation_matrix:
                        correlations[i, j] = self.correlation_matrix[key]
                    elif f"{attr2}_{attr1}" in self.correlation_matrix:
                        correlations[i, j] = self.correlation_matrix[f"{attr2}_{attr1}"]

        # Fit maximum entropy model
        try:
            xs, probs = fit_maxent(marginals, correlations)
        except Exception as e:
            print(f"Failed to fit maxent model: {e}")
            return False

        # Prepare constraint groups and requirements
        groups = []
        r = []

        for constraint in constraints:
            # Find the index of this attribute in our ordered list
            if constraint.attribute in self.attribute_order:
                attr_idx = self.attribute_order.index(constraint.attribute)
                groups.append([attr_idx])
                # Convert min_count to fraction of total capacity
                r.append(constraint.min_count / 1000.0)

        if not groups:
            return False

        # Solve the linear program
        try:
            a = solve_lp(xs, probs, groups, r)
            self.policy = OnlinePolicy(xs, a)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to solve LP: {e}")
            return False

    def should_accept(
        self,
        attributes: Dict[str, bool],
        constraints: List[Constraint],
        current_counts: Dict[str, int],
        admitted: int,
    ) -> bool:
        """
        Determine whether to accept the current person using LP optimization.
        """
        if admitted >= 1000:
            return False

        # Use LP policy if available
        if self.policy:
            return self.policy.accept(attributes)

        # Simple fallback: accept if helps with any constraint
        for constraint in constraints:
            if (
                attributes.get(constraint.attribute, False)
                and current_counts[constraint.attribute] < constraint.min_count
            ):
                return True

        return False

    def get_decision_confidence(
        self,
        attributes: Dict[str, bool],
        constraints: List[Constraint],
        current_counts: Dict[str, int],
        admitted: int,
    ) -> float:
        """
        Get the confidence score for the last decision made.
        """
        if not self.is_initialized or not self.policy:
            return 0.5  # Medium confidence for fallback

        # Convert person attributes to vector
        person_vector = np.zeros(len(self.attribute_order))
        for i, attr in enumerate(self.attribute_order):
            person_vector[i] = 1 if attributes.get(attr, False) else 0

        # Find the acceptance probability for this person type
        for i, x in enumerate(self.policy.xs):
            if np.all(x == person_vector):
                return self.policy.a[i]

        return 0.5  # Default confidence if person type not found
