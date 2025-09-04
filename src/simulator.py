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
    
    def __init__(self, constraints: List[Constraint], total_people: int = 2000, scenario: int = 1):
        self.constraints = constraints
        self.total_people = total_people
        self.people_generated = 0
        self.scenario = scenario
        
        
        if scenario == 1:
            self.distribution = [
                ({}, 1.0),  
            ]
        elif scenario == 2:
            self.distribution = [
                ({}, 63/2341),  
                ({'berlin_local': True}, 136/2341),  
                ({'creative': True}, 3/2341),  
                ({'creative': True, 'berlin_local': True}, 3/2341),  
                ({'well_connected': True}, 72/2341),  
                ({'well_connected': True, 'berlin_local': True}, 614/2341),  
                ({'well_connected': True, 'creative': True}, 4/2341),  
                ({'well_connected': True, 'creative': True, 'berlin_local': True}, 16/2341),  
                ({'techno_lover': True}, 970/2341),  
                ({'techno_lover': True, 'berlin_local': True}, 35/2341),  
                ({'techno_lover': True, 'creative': True}, 13/2341),  
                ({'techno_lover': True, 'creative': True, 'berlin_local': True}, 11/2341),  
                ({'techno_lover': True, 'well_connected': True}, 231/2341),  
                ({'techno_lover': True, 'well_connected': True, 'berlin_local': True}, 85/2341),  
                ({'techno_lover': True, 'well_connected': True, 'creative': True}, 18/2341),  
                ({'techno_lover': True, 'well_connected': True, 'creative': True, 'berlin_local': True}, 67/2341),  
            ]
        elif scenario == 3:
            self.distribution = [
                ({}, 211/16349),  
                ({'german_speaker': True}, 361/16349),  
                ({'vinyl_collector': True}, 3/16349),  
                ({'vinyl_collector': True, 'german_speaker': True}, 13/16349),  
                ({'queer_friendly': True}, 2/16349),  
                ({'queer_friendly': True, 'german_speaker': True}, 4/16349),  
                ({'queer_friendly': True, 'vinyl_collector': True}, 4/16349),  
                ({'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 4/16349),  
                ({'fashion_forward': True}, 218/16349),  
                ({'fashion_forward': True, 'german_speaker': True}, 1024/16349),  
                ({'fashion_forward': True, 'vinyl_collector': True}, 5/16349),  
                ({'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 7/16349),  
                ({'fashion_forward': True, 'queer_friendly': True}, 6/16349),  
                ({'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 16/16349),  
                ({'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 7/16349),  
                ({'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 9/16349),  
                ({'international': True}, 270/16349),  
                ({'international': True, 'german_speaker': True}, 91/16349),  
                ({'international': True, 'vinyl_collector': True}, 3/16349),  
                ({'international': True, 'vinyl_collector': True, 'german_speaker': True}, 2/16349),  
                ({'international': True, 'queer_friendly': True}, 4/16349),  
                ({'international': True, 'queer_friendly': True, 'german_speaker': True}, 3/16349),  
                ({'international': True, 'queer_friendly': True, 'vinyl_collector': True}, 7/16349),  
                ({'international': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 7/16349),  
                ({'international': True, 'fashion_forward': True}, 2576/16349),  
                ({'international': True, 'fashion_forward': True, 'german_speaker': True}, 351/16349),  
                ({'international': True, 'fashion_forward': True, 'vinyl_collector': True}, 13/16349),  
                ({'international': True, 'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 8/16349),  
                ({'international': True, 'fashion_forward': True, 'queer_friendly': True}, 32/16349),  
                ({'international': True, 'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 29/16349),  
                ({'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 12/16349),  
                ({'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 15/16349),  
                ({'underground_veteran': True}, 113/16349),  
                ({'underground_veteran': True, 'german_speaker': True}, 2525/16349),  
                ({'underground_veteran': True, 'vinyl_collector': True}, 31/16349),  
                ({'underground_veteran': True, 'vinyl_collector': True, 'german_speaker': True}, 131/16349),  
                ({'underground_veteran': True, 'queer_friendly': True}, 4/16349),  
                ({'underground_veteran': True, 'queer_friendly': True, 'german_speaker': True}, 27/16349),  
                ({'underground_veteran': True, 'queer_friendly': True, 'vinyl_collector': True}, 14/16349),  
                ({'underground_veteran': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 53/16349),  
                ({'underground_veteran': True, 'fashion_forward': True}, 200/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'german_speaker': True}, 1731/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'vinyl_collector': True}, 6/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 36/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True}, 33/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 41/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 28/16349),  
                ({'underground_veteran': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 39/16349),  
                ({'underground_veteran': True, 'international': True}, 724/16349),  
                ({'underground_veteran': True, 'international': True, 'german_speaker': True}, 226/16349),  
                ({'underground_veteran': True, 'international': True, 'vinyl_collector': True}, 15/16349),  
                ({'underground_veteran': True, 'international': True, 'vinyl_collector': True, 'german_speaker': True}, 53/16349),  
                ({'underground_veteran': True, 'international': True, 'queer_friendly': True}, 24/16349),  
                ({'underground_veteran': True, 'international': True, 'queer_friendly': True, 'german_speaker': True}, 9/16349),  
                ({'underground_veteran': True, 'international': True, 'queer_friendly': True, 'vinyl_collector': True}, 17/16349),  
                ({'underground_veteran': True, 'international': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 30/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True}, 4244/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'german_speaker': True}, 388/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'vinyl_collector': True}, 25/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'vinyl_collector': True, 'german_speaker': True}, 29/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True}, 64/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True, 'german_speaker': True}, 52/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True}, 40/16349),  
                ({'underground_veteran': True, 'international': True, 'fashion_forward': True, 'queer_friendly': True, 'vinyl_collector': True, 'german_speaker': True}, 62/16349),  
            ]
        else:
            raise ValueError(f"Unknown scenario: {scenario}. Supported scenarios: 1, 2, 3")
        
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
    def __init__(self, constraints: List[Constraint], solver, scenario: int = 1):
        self.constraints = constraints
        self.solver = solver
        self.scenario = scenario
        self.simulator = GameSimulator(constraints, total_people=10000, scenario=scenario)  # Large number to ensure we don't run out
        
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


def run_simulation(constraints: List[Constraint], solver, scenario: int = 1) -> Dict:
    game = SimulatedGame(constraints, solver, scenario)
    results = game.play_game()
    game.print_final_statistics(results)
    return results
