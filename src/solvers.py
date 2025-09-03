import time
from abc import ABC, abstractmethod
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from api import BerghainAPIClient, Constraint

class BaseSolver(ABC):
    """Base class for all Berghain solvers."""

    console = Console()
    
    def __init__(self, api_client: BerghainAPIClient = None):
        self.api_client = api_client or BerghainAPIClient()
    
    @abstractmethod
    def should_accept(self, attributes: Dict[str, bool], constraints: List[Constraint],
                     current_counts: Dict[str, int], admitted: int) -> bool:
        """Determine whether to accept or reject a person."""
        pass
    
    def play_game(self, scenario: int, player_id: str) -> int:
        """
        Play a complete game using the solver's strategy.
        """
        # Create game
        game_data = self.api_client.create_game(scenario, player_id)
        game_id = game_data["gameId"]
        constraints = self.api_client.parse_constraints(game_data)

        url = f"https://berghain.challenges.listenlabs.ai/game/{game_id}"
        
        self.console.print(url, style="blue")
        print(f"Constraints: {[(c.attribute, c.min_count) for c in constraints]}")
        print(f"Attribute Statistics:")
        if "attributeStatistics" in game_data:
            stats = game_data["attributeStatistics"]
            if "relativeFrequencies" in stats:
                print(f"  Relative Frequencies: {stats['relativeFrequencies']}")
            if "correlations" in stats:
                print(f"  Correlations: {stats['correlations']}")
        
        # Initialize LP policy if this is an LP solver
        if hasattr(self, 'update_statistics') and hasattr(self, 'initialize_policy'):
            self.update_statistics(game_data)
            if self.initialize_policy(constraints):
                print("✅ LP policy initialized successfully")
            else:
                print("⚠️  LP policy initialization failed, using fallback logic")
        
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
            key = ' + '.join(key_parts) if key_parts else 'none'
            admitted_tracking[key] = 0
            rejected_tracking[key] = 0
        
        # Get first person
        response = self.api_client.get_first_person(game_id)
        
        decision_count = 0
        
        # Initialize progress bar
        capacity = 2000
        pbar = tqdm(total=capacity, desc="Processing people", unit="person", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        while response["status"] == "running":
            person_data = response["nextPerson"]
            person_index = person_data["personIndex"]
            attributes = person_data["attributes"]
            
            admitted = response["admittedCount"]
            rejected = response["rejectedCount"]
            total_processed = admitted + rejected
            
            # Make optimal decision using solver strategy
            accept = self.should_accept(attributes, constraints, current_counts, admitted)
            
            # Submit decision for THIS person and get next person
            response = self.api_client.make_decision(game_id, person_index, accept)
            
            # Track person by characteristics (both admitted and rejected)
            key_parts = []
            for attr in constraint_attributes:
                if attributes.get(attr, False):
                    key_parts.append(attr)
            key = ' + '.join(key_parts) if key_parts else 'none'
            
            # Update counts if accepted
            if accept and response["status"] == "running":
                for attr, value in attributes.items():
                    if value:
                        current_counts[attr] += 1
                admitted_tracking[key] += 1
            else:
                rejected_tracking[key] += 1
            
            decision_count += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Print detailed stats on separate lines every 50 people to avoid clutter
            remaining_capacity = 1000 - admitted

            if decision_count % 50 == 0 or remaining_capacity == 1:
                # Calculate constraint progress
                constraint_progress = []
                for constraint in constraints:
                    current_count = current_counts[constraint.attribute]
                    progress_pct = (current_count / constraint.min_count) * 100
                    constraint_progress.append(f"{constraint.attribute}: {current_count}/{constraint.min_count} ({progress_pct:.1f}%)")
                
                constraint_str = " | ".join(constraint_progress)
                
                # Create and display updated Rich table
                total_admitted = sum(admitted_tracking.values())
                total_rejected = sum(rejected_tracking.values())
                
                # Create a new table for each update
                update_table = Table(title="Statistics", show_header=True, header_style="bold magenta")
                update_table.add_column("Category", style="cyan", no_wrap=True)
                update_table.add_column("Admitted", justify="right", style="green")
                update_table.add_column("Rejected", justify="right", style="red")
                update_table.add_column("Adm %", justify="right", style="yellow")
                update_table.add_column("Rej %", justify="right", style="orange3")
                
                for category in admitted_tracking.keys():
                    admitted_count = admitted_tracking[category]
                    rejected_count = rejected_tracking.get(category, 0)
                    
                    # Calculate percentages
                    admitted_pct = (admitted_count / total_admitted * 100) if total_admitted > 0 else 0
                    rejected_pct = (rejected_count / total_rejected * 100) if total_rejected > 0 else 0
                    
                    update_table.add_row(
                        category,
                        str(admitted_count),
                        str(rejected_count),
                        f"{admitted_pct:.1f}%",
                        f"{rejected_pct:.1f}%"
                    )
                
                # Display the updated table
                print()
                self.console.print(update_table)
                print(f"  Admitted: {admitted}, Rejected: {rejected}, Remaining Capacity: {remaining_capacity}")
                print(f"  Constraints: {constraint_str}")
                pbar.refresh()  # Refresh the progress bar after printing
        
        # Close progress bar
        pbar.close()
        
        if response["status"] == "completed":
            self.console.print(f'Finished game: {url}', style="blue")
            return 0
        else:
            self.console.print(f'Game ended with status {response['status']}: {url}', style="red")
            return response.get("rejectedCount", 99999)



# Example of how to create additional solvers
class GreedySolver(BaseSolver):
    """Simple greedy solver that accepts anyone with positive attributes."""
    
    def should_accept(self, attributes: Dict[str, bool], constraints: List[Constraint],
                     current_counts: Dict[str, int], admitted: int) -> bool:
        """Accept anyone with at least one positive attribute."""
        if admitted >= 1000:
            return False
        
        positive_count = sum(1 for attr, value in attributes.items() if value)
        return positive_count > 0


class ConservativeSolver(BaseSolver):
    """Conservative solver that only accepts people with multiple positive attributes."""
    
    def should_accept(self, attributes: Dict[str, bool], constraints: List[Constraint],
                     current_counts: Dict[str, int], admitted: int) -> bool:
        """Only accept people with multiple positive attributes."""
        if admitted >= 1000:
            return False
        
        positive_count = sum(1 for attr, value in attributes.items() if value)
        return positive_count >= 2
