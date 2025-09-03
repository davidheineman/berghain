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
        self.constraints = constraints
        self.total_people = total_people
        self.people_generated = 0
        
        # Define the distribution based on the provided statistics
        # This represents the probability of each attribute combination
        self.distribution = [
            # Format: (attributes_dict, probability)
            ({}, 63/2341),  # none: 63/2341 total people
            ({'berlin_local': True}, 136/2341),  # berlin_local: 136/2341 (30+106)
            ({'creative': True}, 3/2341),  # creative: 3/2341 (2+1)
            ({'creative': True, 'berlin_local': True}, 3/2341),  # creative + berlin_local: 3/2341 (3+0)
            ({'well_connected': True}, 72/2341),  # well_connected: 72/2341 (21+51)
            ({'well_connected': True, 'berlin_local': True}, 614/2341),  # well_connected + berlin_local: 614/2341 (297+317)
            ({'well_connected': True, 'creative': True}, 4/2341),  # well_connected + creative: 4/2341 (4+0)
            ({'well_connected': True, 'creative': True, 'berlin_local': True}, 16/2341),  # well_connected + creative + berlin_local: 16/2341 (16+0)
            ({'techno_lover': True}, 970/2341),  # techno_lover: 970/2341 (283+687)
            ({'techno_lover': True, 'berlin_local': True}, 35/2341),  # techno_lover + berlin_local: 35/2341 (23+12)
            ({'techno_lover': True, 'creative': True}, 13/2341),  # techno_lover + creative: 13/2341 (13+0)
            ({'techno_lover': True, 'creative': True, 'berlin_local': True}, 11/2341),  # techno_lover + creative + berlin_local: 11/2341 (11+0)
            ({'techno_lover': True, 'well_connected': True}, 231/2341),  # techno_lover + well_connected: 231/2341 (138+93)
            ({'techno_lover': True, 'well_connected': True, 'berlin_local': True}, 85/2341),  # techno_lover + well_connected + berlin_local: 85/2341 (73+12)
            ({'techno_lover': True, 'well_connected': True, 'creative': True}, 18/2341),  # techno_lover + well_connected + creative: 18/2341 (18+0)
            ({'techno_lover': True, 'well_connected': True, 'creative': True, 'berlin_local': True}, 67/2341),  # techno_lover + well_connected + creative + berlin_local: 67/2341 (67+0)
        ]
        
        # Normalize the distribution to sum to 1
        total_prob = sum(prob for _, prob in self.distribution)
        self.normalized_distribution = [
            (attrs, prob / total_prob) for attrs, prob in self.distribution
        ]
        
        # Extract attributes and probabilities for numpy sampling
        self.attributes_list = [attrs for attrs, _ in self.normalized_distribution]
        self.probabilities = [prob for _, prob in self.normalized_distribution]
    
    def generate_person(self) -> Person:
        if self.people_generated >= self.total_people:
            raise StopIteration("No more people to generate")
        
        # Sample from the distribution using numpy
        index = np.random.choice(len(self.attributes_list), p=self.probabilities)
        attrs = self.attributes_list[index]
        
        self.people_generated += 1
        return Person(attributes=attrs.copy())
    
    def get_next_person(self) -> Optional[Person]:
        try:
            return self.generate_person()
        except StopIteration:
            return None
    
    def reset(self):
        self.people_generated = 0


class SimulatedGame:
    def __init__(self, constraints: List[Constraint], solver):
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
