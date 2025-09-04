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

    # Decision variables: a_t for each type t (0..T-1) and y = total expected admissions fraction
    # Objective: maximize y (equivalently minimize -y)
    c = np.zeros(T + 1)
    c[-1] = -1.0

    # Inequality constraints: for each group G with required fraction rk among admitted
    #   sum_t mask_t * probs_t * a_t - rk * y >= 0  ->  -sum(...) + rk*y <= 0
    A_ub = []
    b_ub = []
    for G, rk in zip(groups, r):
        mask = np.array([int(all(x[i] == 1 for i in G)) for x in xs], dtype=float)
        row = np.zeros(T + 1)
        row[:T] = -mask * probs
        row[-1] = rk
        A_ub.append(row)
        b_ub.append(0.0)

    # Equality constraint: y = sum_t probs_t * a_t  ->  sum(probs*a) - y = 0
    A_eq = np.zeros((1, T + 1))
    A_eq[0, :T] = probs
    A_eq[0, -1] = -1.0
    b_eq = np.array([0.0])

    # Bounds: 0 <= a_t <= 1, 0 <= y <= 1
    bounds = [(0.0, 1.0)] * T + [(0.0, 1.0)]

    res = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if A_ub else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if res.success:
        return res.x[:T]
    else:
        # Infeasible: fall back to prioritizing constraint-satisfying types
        fallback_probs = np.ones(T) * 0.05
        for G, rk in zip(groups, r):
            mask = np.array([int(all(x[i] == 1 for i in G)) for x in xs], dtype=float)
            fallback_probs += mask * 0.5
        return np.clip(fallback_probs, 0.0, 1.0)


class OnlinePolicy:
    """Online policy for making acceptance decisions based on pre-computed probabilities."""

    def __init__(self, xs, a, probs=None, deterministic=False):
        self.xs = xs
        self.a = a
        self.probs = probs
        self.deterministic = deterministic and probs is not None

        # Quota-based deterministic acceptance to reduce variance when distribution is known
        self.quotas_remaining = None
        if self.deterministic:
            admitted_weights = self.probs * self.a
            y = admitted_weights.sum()
            if y > 0:
                fractions = (
                    admitted_weights / y
                )  # fraction among admitted for each type
                raw_targets = 1000.0 * fractions
                base = np.floor(raw_targets).astype(int)
                remainder = 1000 - int(base.sum())
                # Distribute remaining by largest fractional parts
                fractional_parts = raw_targets - base
                if remainder > 0:
                    top_indices = np.argsort(-fractional_parts)[:remainder]
                    base[top_indices] += 1
                self.quotas_remaining = base
            else:
                # Fallback to probabilistic if degenerate
                self.deterministic = False

    def accept(self, person):
        """Determine whether to accept a person based on their attributes."""
        # Find type index
        idx = None
        for i, x in enumerate(self.xs):
            if np.all(x == person):
                idx = i
                break

        if idx is None:
            return False

        if self.deterministic and self.quotas_remaining is not None:
            if self.quotas_remaining[idx] > 0:
                self.quotas_remaining[idx] -= 1
                return True
            else:
                return False

        # Probabilistic fallback
        prob = self.a[idx]
        accepted = np.random.rand() < prob
        return accepted


class LinearProgrammingSolver(BaseSolver):
    """
    Linear Programming-based solver that optimizes acceptance decisions.

    This solver uses a maximum entropy model to estimate the distribution of future people
    and solves a linear program to find optimal acceptance probabilities that satisfy
    all constraints with high probability.
    """

    def __init__(
        self,
        attribute_frequencies,
        correlation_matrix,
        constraints,
        distribution,
        api_client=None,
    ):
        """
        Initialize the LP solver.

        Args:
            api_client: API client for the game
        """
        super().__init__(api_client)
        self.attribute_frequencies = attribute_frequencies
        self.correlation_matrix = correlation_matrix
        self.distribution = distribution
        self.constraints = constraints
        self.attribute_order = []
        self.marginals = None
        self.priority_attrs = set()
        self.policy = None
        self.is_initialized = False

        self.initialize_policy(constraints)

    def initialize_policy(self, constraints: List[Constraint]):
        """
        Initialize the LP policy based on current statistics and constraints.
        This should be called once we have enough data about attribute frequencies.
        """
        xs = None
        probs = None

        # Build attribute universe
        all_attributes = set(c.attribute for c in constraints)

        # If a full distribution is provided, use it directly for xs and probs
        if self.distribution:
            # Collect attributes that appear in the distribution
            for conditions, _p in self.distribution:
                for attr in conditions.keys():
                    all_attributes.add(attr)

            # Also include any attributes we have statistics for (safety)
            if self.attribute_frequencies:
                all_attributes.update(self.attribute_frequencies.keys())

            self.attribute_order = sorted(list(all_attributes))
            d = len(self.attribute_order)
            if d == 0:
                return False

            # Build xs and probs from the provided distribution
            xs_list = []
            probs_list = []
            for conditions, p in self.distribution:
                x_vec = np.zeros(d)
                for i, attr in enumerate(self.attribute_order):
                    x_vec[i] = 1 if conditions.get(attr, False) else 0
                xs_list.append(x_vec)
                probs_list.append(float(p))

            xs = np.array(xs_list)
            probs = np.array(probs_list, dtype=float)

            # Normalize probabilities in case of tiny numerical issues
            total_p = probs.sum()
            if total_p > 0:
                probs = probs / total_p

            # Compute marginals from the explicit distribution for feasibility checks
            marginals = (probs[:, None] * xs).sum(axis=0)
            self.marginals = marginals

        else:
            # No explicit distribution; fall back to maxent fit using frequencies and correlations
            if not self.attribute_frequencies or not self.correlation_matrix:
                return False

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
                        # Handle nested correlation matrix format
                        if (
                            isinstance(self.correlation_matrix, dict)
                            and attr1 in self.correlation_matrix
                        ):
                            if (
                                isinstance(self.correlation_matrix[attr1], dict)
                                and attr2 in self.correlation_matrix[attr1]
                            ):
                                correlations[i, j] = self.correlation_matrix[attr1][
                                    attr2
                                ]

            # Fit maximum entropy model
            try:
                xs, probs = fit_maxent(marginals, correlations)
            except Exception as e:
                print(f"Failed to fit maxent model: {e}")
                return False
            self.marginals = marginals

        # Prepare constraint groups and requirements
        groups = []
        r = []

        for constraint in constraints:
            # Find the index of this attribute in our ordered list
            if constraint.attribute in self.attribute_order:
                attr_idx = self.attribute_order.index(constraint.attribute)
                groups.append([attr_idx])

                required_fraction = constraint.min_count / 1000.0

                if self.distribution:
                    # With explicit distribution, add a small cushion to counter sampling variance
                    safety_margin = 0.02
                    r.append(min(0.99, required_fraction + safety_margin))
                else:
                    # Feasibility adjustment only in the estimated-distribution mode
                    available_fraction = marginals[attr_idx]
                    if available_fraction >= required_fraction:
                        r.append(required_fraction)
                    else:
                        adjusted_fraction = available_fraction * 0.5
                        r.append(adjusted_fraction)
                        adjusted_target = int(adjusted_fraction * 1000)
                        print(
                            f"Warning: {constraint.attribute} constraint infeasible. "
                            f"Required: {required_fraction:.3f}, Available: {available_fraction:.3f}. "
                            f"Adjusting to: {adjusted_fraction:.3f} (target: {adjusted_target})"
                        )

        # Identify severely scarce attributes to prioritize when distribution is available
        if self.distribution and self.marginals is not None:
            scarcity_threshold = 3.0
            for constraint in constraints:
                if constraint.attribute in self.attribute_order:
                    idx = self.attribute_order.index(constraint.attribute)
                    req = constraint.min_count / 1000.0
                    avail = (
                        float(self.marginals[idx])
                        if self.marginals is not None
                        else 0.0
                    )
                    if avail > 0 and (req / max(avail, 1e-9)) >= scarcity_threshold:
                        self.priority_attrs.add(constraint.attribute)

        if not groups:
            return False

        # Solve the linear program
        try:
            a = solve_lp(xs, probs, groups, r)
            # Use probabilistic policy by default; deterministic quotas can be too brittle in finite samples
            self.policy = OnlinePolicy(xs, a, probs=probs, deterministic=False)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to solve LP: {e}")
            return False

    def should_accept(
        self,
        attributes: Dict[str, bool],
        current_counts: Dict[str, int],
        admitted: int,
    ) -> bool:
        """
        Determine whether to accept the current person using LP optimization.
        """
        if admitted >= 1000:
            return False

        # Ensure required statistics are set
        if not (
            (self.attribute_frequencies and self.correlation_matrix)
            or self.distribution
        ):
            raise ValueError(
                "LP solver requires either a full distribution or attribute statistics (frequencies and correlations)."
            )

        # Capacity-aware logic with aggressive fallback rule
        remaining_capacity = 1000 - admitted

        # Calculate how many people we still need for each constraint (using ORIGINAL targets)
        still_needed = {}
        for constraint in self.constraints:
            still_needed[constraint.attribute] = max(
                0, constraint.min_count - current_counts[constraint.attribute]
            )

        # Check if we're in a critical situation where we need to be very selective
        # This happens when we have limited capacity left and multiple constraints are still unmet
        unmet_constraints = [
            attr for attr, needed in still_needed.items() if needed > 0
        ]

        # FINAL-PUSH fallback: disable when using explicit distribution (LP already enforces ratios)
        if len(unmet_constraints) >= 2 and not self.distribution:
            # From the very beginning: require at least 2 unmet constraints
            unmet_attributes_satisfied = 0
            for constraint in self.constraints:
                if still_needed[constraint.attribute] > 0:
                    if attributes.get(constraint.attribute, False):
                        unmet_attributes_satisfied += 1

            if remaining_capacity <= 950:
                # Very early: require at least 2 unmet constraints
                if unmet_attributes_satisfied >= 2:
                    return True
                else:
                    return False

            elif remaining_capacity <= 700:
                # Early-mid game: require at least 3 unmet constraints
                if unmet_attributes_satisfied >= 3:
                    return True
                else:
                    return False

            elif remaining_capacity <= 400:
                # Mid-late game: require ALL unmet constraints to be satisfied
                if unmet_attributes_satisfied == len(unmet_constraints):
                    return True
                else:
                    return False

        # Use LP policy if available
        if self.policy:
            # Convert attributes dictionary to vector format
            person_vector = np.zeros(len(self.attribute_order))
            for i, attr in enumerate(self.attribute_order):
                person_vector[i] = 1 if attributes.get(attr, False) else 0
            return self.policy.accept(person_vector)

        # Simple fallback: accept if helps with any constraint
        for constraint in self.constraints:
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
