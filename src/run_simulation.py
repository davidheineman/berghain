import argparse
from typing import Dict, List
from simulator import SimulatedGame
from lp_solver import LinearProgrammingSolver
from constants import get_constraints, get_corr, get_frequencies
from api import Constraint


def run_simulation(constraints: List[Constraint], solver, scenario: int = 1) -> Dict:
    game = SimulatedGame(constraints, solver, scenario)
    results = game.play_game()
    return results


def run_multiple_simulations(constraints, solver, num_runs=10, scenario=1):
    all_results = []
    successful_runs = 0
    
    for i in range(num_runs):
        result = run_simulation(constraints, solver, scenario)
        all_results.append(result)
        
        if result['success']:
            successful_runs += 1
    
    # Calculate averages
    avg_results = {
        'success_rate': successful_runs / num_runs,
        'avg_admitted': sum(r['admitted'] for r in all_results) / num_runs,
        'avg_rejected': sum(r['rejected'] for r in all_results) / num_runs,
        'avg_constraint_satisfaction': {}
    }
    for constraint in constraints:
        attr = constraint.attribute
        avg_satisfied = sum(r['constraint_status'][attr]['current'] for r in all_results) / num_runs
        avg_results['avg_constraint_satisfaction'][attr] = {
            'satisfied': avg_satisfied,
            'required': constraint.min_count,
            'percentage': avg_satisfied / constraint.min_count * 100
        }
    
    # Print results
    print(f"Success Rate: {successful_runs}/{num_runs} ({avg_results['success_rate']*100:.1f}%)")
    print(f"Average Admitted: {avg_results['avg_admitted']:.0f}/1000")
    print(f"Average Rejected: {avg_results['avg_rejected']:.0f}")
    print("\nAverage Constraint Satisfaction:")
    for attr, stats in avg_results['avg_constraint_satisfaction'].items():
        if stats['satisfied'] >= stats['required']:
            status = "\033[92m✓\033[0m"  # Green checkmark
            color = "\033[92m"  # Green
        else:
            status = "\033[91m✗\033[0m"  # Red X
            color = "\033[91m"  # Red
        print(f"  {attr}: {color}{stats['satisfied']:.0f}/{stats['required']} ({stats['percentage']:.1f}%)\033[0m {status}")
    
    return avg_results

def main():
    parser = argparse.ArgumentParser(description='Run Berghain game simulation')
    parser.add_argument('--solver', choices=['lp'], default='lp', help='Solver to use')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3], default=1, help='Scenario to run (1, 2, or 3)')
    
    args = parser.parse_args()

    constraints = get_constraints(scenario=args.scenario)
    
    solver = LinearProgrammingSolver(
        attribute_frequencies=get_frequencies(scenario=args.scenario),
        correlation_matrix=get_corr(scenario=args.scenario),
        constraints=constraints,
    )

    return run_multiple_simulations(constraints, solver, args.runs, args.scenario)


if __name__ == "__main__":
    main()
