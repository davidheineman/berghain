import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from api import Constraint


@dataclass
class Person:
    """Represents a person with their attributes."""
    attributes: Dict[str, bool]
    
    def has_attribute(self, attr: str) -> bool:
        return self.attributes.get(attr, False)


class GameSimulator:
    """
    Simulator for the Berghain game that generates people based on a given distribution.
    """
    
    def __init__(self, constraints: List[Constraint], total_people: int = 2000):
        """
        Initialize the simulator.
        
        Args:
            constraints: List of constraints for the game
            total_people: Total number of people to generate
        """
        self.constraints = constraints
        self.total_people = total_people
        self.people_generated = 0
        
        # Define the distribution based on the provided statistics
        # This represents the probability of each attribute combination
        self.distribution = [
            # Format: (attributes_dict, probability)
            ({}, 0.040),  # none: 4.0% (40/1058 total people)
            ({'berlin_local': True}, 0.056),  # berlin_local: 5.6% (59/1058)
            ({'creative': True}, 0.000),  # creative: 0.0% (0/1058)
            ({'creative': True, 'berlin_local': True}, 0.001),  # creative + berlin_local: 0.1% (1/1058)
            ({'well_connected': True}, 0.035),  # well_connected: 3.5% (37/1058)
            ({'well_connected': True, 'berlin_local': True}, 0.234),  # well_connected + berlin_local: 23.4% (248/1058)
            ({'well_connected': True, 'creative': True}, 0.003),  # well_connected + creative: 0.3% (3/1058)
            ({'well_connected': True, 'creative': True, 'berlin_local': True}, 0.002),  # well_connected + creative + berlin_local: 0.2% (2/1058)
            ({'techno_lover': True}, 0.424),  # techno_lover: 42.4% (448/1058)
            ({'techno_lover': True, 'berlin_local': True}, 0.012),  # techno_lover + berlin_local: 1.2% (13/1058)
            ({'techno_lover': True, 'creative': True}, 0.005),  # techno_lover + creative: 0.5% (5/1058)
            ({'techno_lover': True, 'creative': True, 'berlin_local': True}, 0.008),  # techno_lover + creative + berlin_local: 0.8% (8/1058)
            ({'techno_lover': True, 'well_connected': True}, 0.100),  # techno_lover + well_connected: 10.0% (106/1058)
            ({'techno_lover': True, 'well_connected': True, 'berlin_local': True}, 0.048),  # techno_lover + well_connected + berlin_local: 4.8% (51/1058)
            ({'techno_lover': True, 'well_connected': True, 'creative': True}, 0.010),  # techno_lover + well_connected + creative: 1.0% (11/1058)
            ({'techno_lover': True, 'well_connected': True, 'creative': True, 'berlin_local': True}, 0.024),  # techno_lover + well_connected + creative + berlin_local: 2.4% (25/1058)
        ]
        
        # Normalize the distribution to sum to 1
        total_prob = sum(prob for _, prob in self.distribution)
        self.normalized_distribution = [
            (attrs, prob / total_prob) for attrs, prob in self.distribution
        ]
        
        # Create cumulative distribution for sampling
        self.cumulative_dist = []
        cumsum = 0.0
        for attrs, prob in self.normalized_distribution:
            cumsum += prob
            self.cumulative_dist.append((attrs, cumsum))
    
    def generate_person(self) -> Person:
        """
        Generate a single person based on the distribution.
        
        Returns:
            Person object with attributes
        """
        if self.people_generated >= self.total_people:
            raise StopIteration("No more people to generate")
        
        # Sample from the distribution
        rand_val = random.random()
        for attrs, cumsum in self.cumulative_dist:
            if rand_val <= cumsum:
                self.people_generated += 1
                return Person(attributes=attrs.copy())
        
        # Fallback (should never happen)
        self.people_generated += 1
        return Person(attributes={})
    
    def get_next_person(self) -> Optional[Person]:
        """
        Get the next person in the sequence.
        
        Returns:
            Person object or None if no more people
        """
        try:
            return self.generate_person()
        except StopIteration:
            return None
    
    def reset(self):
        """Reset the simulator to start over."""
        self.people_generated = 0
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the distribution.
        
        Returns:
            Dictionary with distribution statistics
        """
        return {
            'total_combinations': len(self.distribution),
            'normalized_distribution': self.normalized_distribution,
            'people_generated': self.people_generated,
            'total_people': self.total_people
        }


class SimulatedGame:
    """
    Simulated version of the Berghain game that uses the simulator.
    """
    
    def __init__(self, constraints: List[Constraint], solver):
        """
        Initialize the simulated game.
        
        Args:
            constraints: List of constraints for the game
            solver: The solver to use for decisions
        """
        self.constraints = constraints
        self.solver = solver
        self.simulator = GameSimulator(constraints, total_people=10000)  # Large number to ensure we don't run out
        
        # Game state
        self.admitted = 0
        self.rejected = 0
        self.current_counts = {constraint.attribute: 0 for constraint in constraints}
        self.admitted_by_category = {}
        self.rejected_by_category = {}
        
        # Statistics
        self.decisions = []
    
    def play_game(self) -> Dict:
        # Process people until we admit exactly 1000
        person_count = 0
        while self.admitted < 1000 and (self.admitted + self.rejected) < 10_000:
            person = self.simulator.get_next_person()
            if person is None:
                # If we run out of people from the simulator, generate more
                # Reset the simulator to continue generating people
                self.simulator.reset()
                person = self.simulator.get_next_person()
                if person is None:
                    print("ERROR: Simulator failed to generate people!")
                    break
            
            person_count += 1
            
            # Make decision
            should_accept = self.solver.should_accept(
                person.attributes, self.constraints, self.current_counts, self.admitted
            )
            
            # Record decision
            decision = {
                'person_id': person_count,
                'attributes': person.attributes.copy(),
                'accepted': should_accept,
                'admitted_count': self.admitted,
                'current_counts': self.current_counts.copy()
            }
            self.decisions.append(decision)
            
            if should_accept:
                self.admitted += 1
                # Update counts for admitted person
                for attr, value in person.attributes.items():
                    if value:
                        self.current_counts[attr] = self.current_counts.get(attr, 0) + 1
                
                # Update category statistics
                category = self._get_category(person.attributes)
                self.admitted_by_category[category] = self.admitted_by_category.get(category, 0) + 1
            else:
                self.rejected += 1
                # Update category statistics
                category = self._get_category(person.attributes)
                self.rejected_by_category[category] = self.rejected_by_category.get(category, 0) + 1

        # Final results
        return self._get_final_results()
    
    def _get_category(self, attributes: Dict[str, bool]) -> str:
        """Get the category string for a person's attributes."""
        if not attributes:
            return "none"
        
        attrs = [attr for attr, value in attributes.items() if value]
        return " + ".join(sorted(attrs))
    
    def _get_final_results(self) -> Dict:
        """Get final game results."""
        # Check if all constraints are satisfied
        all_satisfied = all(
            self.current_counts[constraint.attribute] >= constraint.min_count
            for constraint in self.constraints
        )
        
        # Calculate success metrics
        capacity_filled = self.admitted >= 1000
        success = all_satisfied and capacity_filled
        
        return {
            'success': success,
            'admitted': self.admitted,
            'rejected': self.rejected,
            'capacity_filled': capacity_filled,
            'all_constraints_satisfied': all_satisfied,
            'constraint_status': {
                constraint.attribute: {
                    'current': self.current_counts[constraint.attribute],
                    'needed': constraint.min_count,
                    'satisfied': self.current_counts[constraint.attribute] >= constraint.min_count
                }
                for constraint in self.constraints
            },
            'admitted_by_category': self.admitted_by_category,
            'rejected_by_category': self.rejected_by_category,
            'decisions': self.decisions
        }
    
    def print_final_statistics(self, results: Dict):
        """Print final game statistics."""
        print("=" * 60)
        print("FINAL GAME RESULTS")
        print("=" * 60)
        print(f"Success: {'YES' if results['success'] else 'NO'}")
        print(f"Admitted: {results['admitted']}/1000")
        print(f"Rejected: {results['rejected']}")
        print(f"Capacity Filled: {'YES' if results['capacity_filled'] else 'NO'}")
        print(f"All Constraints Satisfied: {'YES' if results['all_constraints_satisfied'] else 'NO'}")
        print()
        
        print("CONSTRAINT STATUS:")
        for attr, status in results['constraint_status'].items():
            print(f"  {attr}: {status['current']}/{status['needed']} ({'✓' if status['satisfied'] else '✗'})")
        print()
        
        print("ADMITTED BY CATEGORY:")
        for category, count in sorted(results['admitted_by_category'].items()):
            print(f"  {category}: {count}")
        print()
        
        print("REJECTED BY CATEGORY:")
        for category, count in sorted(results['rejected_by_category'].items()):
            print(f"  {category}: {count}")


def run_simulation(constraints: List[Constraint], solver) -> Dict:
    game = SimulatedGame(constraints, solver)
    results = game.play_game()
    game.print_final_statistics(results)
    return results
