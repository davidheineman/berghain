import time
import argparse
from typing import List
from solvers import GreedySolver, ConservativeSolver
from lp_solver import LinearProgrammingSolver
from constants import get_constraints, get_corr, get_frequencies

def run_trials(scenario: int, player_id: str, solver_class, num_trials: int = 3) -> List[int]:
    """Run multiple trials with rate limit handling."""
    solver = solver_class()
    results = []
    
    for trial in range(num_trials):        
        start_time = time.time()
        rejections = solver.play_game(scenario, player_id)
        duration = time.time() - start_time
        
        if rejections < 99999:
            results.append(rejections)
            print(f"  ✅ Result: {rejections} rejections ({duration:.1f}s)")
        else:
            print(f"  ❌ Failed ({duration:.1f}s)")
        
        # Pause to avoid rate limits
        if trial < num_trials - 1:
            time.sleep(5)
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Berghain Challenge Solver')
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3], default=1,
                       help='Scenario to run (1, 2, or 3)')
    parser.add_argument('--player-id', type=str, 
                       default='e44d1e27-daad-4003-bc7d-fbd97d992269',
                       help='Player ID for the game')
    parser.add_argument('--trials', type=int, default=1,
                       help='Number of trials to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all scenarios')
    parser.add_argument('--solver', type=str, choices=['greedy', 'conservative', 'lp'], 
                       default='lp', help='Solver strategy to use')
    
    args = parser.parse_args()
    
    # Select solver class
    solver_classes = {
        'greedy': GreedySolver,
        'conservative': ConservativeSolver,
        'lp': LinearProgrammingSolver
    }
    solver_class = solver_classes[args.solver]
    
    # Determine scenarios to run
    scenarios_to_run = [1, 2, 3] if args.all else [args.scenario]
    num_trials = 2 if args.all else args.trials
    
    all_results = {}
    
    for scenario in scenarios_to_run:
        if args.all:
            print(f"\n{'='*60}\nScenario {scenario}\n{'='*60}")
        
        if num_trials > 1:
            results = run_trials(scenario, args.player_id, solver_class, num_trials)
            all_results[scenario] = results
        else:
            constraints = get_constraints(scenario=scenario)
            solver = solver_class(
                attribute_frequencies=get_frequencies(scenario=scenario),
                correlation_matrix=get_corr(scenario=scenario),
                constraints=constraints,
            )
            result = solver.play_game(scenario, args.player_id)
            print(f"Result: {result} rejections")

if __name__ == "__main__":
    main()
