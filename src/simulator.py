import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from api import Constraint
from constants import get_distribution


@dataclass
class Person:
    attributes: Dict[str, bool]
    
    def has_attribute(self, attr: str) -> bool:
        return self.attributes.get(attr, False)


class GameSimulator:
    def __init__(self, constraints: List[Constraint], total_people: int = 2000, scenario: int = 1):
        self.constraints = constraints
        self.total_people = total_people
        self.people_generated = 0
        self.scenario = scenario
        
        self.distribution = get_distribution(scenario)
        
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
                person.attributes, self.current_counts, self.admitted
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
        if not attributes:
            return "none"
        
        attrs = [attr for attr, value in attributes.items() if value]
        return " + ".join(sorted(attrs))
    
    def _get_final_results(self) -> Dict:
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
