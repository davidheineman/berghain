from abc import ABC, abstractmethod
from typing import Dict
from collections import defaultdict
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from api import BerghainAPIClient


class BaseSolver(ABC):
    console = Console()

    def __init__(self, api_client: BerghainAPIClient = None):
        self.api_client = api_client or BerghainAPIClient()

    @abstractmethod
    def should_accept(
        self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int
    ) -> bool:
        pass

    def play_game(self, scenario: int, player_id: str, verbose: bool = True) -> int:
        # Create game
        game_data = self.api_client.create_game(scenario, player_id)
        game_id = game_data["gameId"]
        constraints = self.api_client.parse_constraints(game_data)

        url = f"https://berghain.challenges.listenlabs.ai/game/{game_id}"

        # Initialize constraints
        stats = game_data["attributeStatistics"]
        self.attribute_frequencies = stats["relativeFrequencies"]
        self.correlation_matrix = stats["correlations"]
        self.initialize_policy(constraints)

        self.console.print(url, style="blue")
        if verbose:
            print(f"Constraints: {[(c.attribute, c.min_count) for c in constraints]}")
            print(f"Attribute Statistics:")
            print(f"  Relative Frequencies: {self.attribute_frequencies}")
            print(f"  Correlations: {self.correlation_matrix}")

        # Track state
        current_counts = defaultdict(int)

        # Track admitted and rejected people by characteristics - dynamically create based on constraints
        constraint_attributes = [constraint.attribute for constraint in constraints]
        admitted_tracking = defaultdict(int)
        rejected_tracking = defaultdict(int)

        # Generate all possible combinations of constraint attributes
        from itertools import product

        for combination in product([False, True], repeat=len(constraint_attributes)):
            # Create a key representing this combination
            key_parts = []
            for i, attr in enumerate(constraint_attributes):
                if combination[i]:
                    key_parts.append(attr)
            key = " + ".join(key_parts) if key_parts else "none"
            admitted_tracking[key] = 0
            rejected_tracking[key] = 0

        # Get first person
        response = self.api_client.get_first_person(game_id)

        decision_count = 0

        # Initialize progress bar
        pbar = tqdm(
            total=1000,
            desc="Filling da club",
            unit="person",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        while response["status"] == "running":
            person_data = response["nextPerson"]
            person_index = person_data["personIndex"]
            attributes = person_data["attributes"]
            admitted = response["admittedCount"]
            rejected = response["rejectedCount"]

            # Make optimal decision using solver strategy
            accept = self.should_accept(attributes, current_counts, admitted)

            # Submit decision for THIS person and get next person
            response = self.api_client.make_decision(game_id, person_index, accept)

            # Track person by characteristics (both admitted and rejected)
            key_parts = []
            for attr in constraint_attributes:
                if attributes.get(attr, False):
                    key_parts.append(attr)
            key = " + ".join(key_parts) if key_parts else "none"

            # Update counts if accepted
            if accept and response["status"] == "running":
                for attr, value in attributes.items():
                    if value:
                        current_counts[attr] += 1
                admitted_tracking[key] += 1
            else:
                rejected_tracking[key] += 1

            decision_count += 1

            # Update progress bar with admitted count
            pbar.n = admitted
            pbar.refresh()

            # Print detailed stats on separate lines every 50 people to avoid clutter
            remaining_capacity = 1000 - admitted

            if verbose and (decision_count % 50 == 0 or remaining_capacity == 1):
                self._print_progress_update(
                    constraints=constraints,
                    current_counts=current_counts,
                    admitted_tracking=admitted_tracking,
                    rejected_tracking=rejected_tracking,
                    url=url,
                    admitted=admitted,
                    rejected=rejected,
                    pbar=pbar,
                )

        # Close progress bar
        pbar.close()

        if response["status"] == "completed":
            self.console.print(f"Finished game: {url}", style="blue")
            return rejected
        else:
            self.console.print(
                f"Game ended with status {response['status']}: {url}", style="red"
            )
            return float('-inf')

    def _print_progress_update(
        self,
        constraints,
        current_counts,
        admitted_tracking,
        rejected_tracking,
        url,
        admitted,
        rejected,
        pbar,
    ):
        # Calculate constraint progress
        constraint_progress = []
        for constraint in constraints:
            current_count = current_counts[constraint.attribute]
            progress_pct = (current_count / constraint.min_count) * 100
            constraint_progress.append(
                f"{constraint.attribute}: {current_count}/{constraint.min_count} ({progress_pct:.1f}%)"
            )

        constraint_str = " | ".join(constraint_progress)

        # Create and display updated Rich table
        total_admitted = sum(admitted_tracking.values())
        total_rejected = sum(rejected_tracking.values())

        # Create a new table for each update
        update_table = Table(
            title="Statistics", show_header=True, header_style="bold magenta"
        )
        update_table.add_column("Category", style="cyan", no_wrap=True)
        update_table.add_column("Admitted", justify="right", style="green")
        update_table.add_column("Rejected", justify="right", style="red")
        update_table.add_column("Adm %", justify="right", style="yellow")
        update_table.add_column("Rej %", justify="right", style="orange3")

        for category in admitted_tracking.keys():
            admitted_count = admitted_tracking[category]
            rejected_count = rejected_tracking.get(category, 0)

            # Calculate percentages
            admitted_pct = (
                (admitted_count / total_admitted * 100) if total_admitted > 0 else 0
            )
            rejected_pct = (
                (rejected_count / total_rejected * 100) if total_rejected > 0 else 0
            )

            update_table.add_row(
                category,
                str(admitted_count),
                str(rejected_count),
                f"{admitted_pct:.1f}%",
                f"{rejected_pct:.1f}%",
            )

        # Display the updated table
        print()
        self.console.print(update_table)
        self.console.print(url, style="blue")
        remaining_capacity = 1000 - admitted
        print(
            f"  Admitted: {admitted}, Rejected: {rejected}, Remaining Capacity: {remaining_capacity}"
        )
        print(f"  Constraints: {constraint_str}")
        pbar.refresh()  # Refresh the progress bar after printing


# Example of how to create additional solvers
class GreedySolver(BaseSolver):
    def should_accept(
        self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int
    ) -> bool:
        if admitted >= 1000:
            return False

        positive_count = sum(1 for attr, value in attributes.items() if value)
        return positive_count > 0


class ConservativeSolver(BaseSolver):
    def should_accept(
        self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int
    ) -> bool:
        if admitted >= 1000:
            return False

        positive_count = sum(1 for attr, value in attributes.items() if value)
        return positive_count >= 2


class RejectAllSolver(BaseSolver):
    def initialize_policy(self, constraints) -> None:
        # No initialization needed for a trivial always-reject policy
        return None

    def should_accept(
        self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int
    ) -> bool:
        # Always reject
        return False


class RuleBasedScenario1Solver(BaseSolver):
    def initialize_policy(self, constraints) -> None:
        return None

    def should_accept(
        self, attributes: Dict[str, bool], current_counts: Dict[str, int], admitted: int
    ) -> bool:
        if admitted >= 1000:
            return False
        N = 1000
        m = {"young": 600, "well_dressed": 600}
        R = N - admitted
        for a in ("young", "well_dressed"):
            if current_counts.get(a, 0) + max(R - 1, 0) < m[a]:
                return bool(attributes.get(a, False))
        return bool(attributes.get("young", False) or attributes.get("well_dressed", False))
