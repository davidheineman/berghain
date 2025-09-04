import argparse
from lp_solver import LinearProgrammingSolver
from constants import get_constraints, get_corr, get_distribution, get_frequencies
from dual_solver import DualThresholdSolver

DEFAULT_PLAYER_ID = "e44d1e27-daad-4003-bc7d-fbd97d992269"

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def run_single_trial(solver, scenario, player_id):
    """Run a single trial of the game."""
    return solver.play_game(scenario, player_id, verbose=False)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Berghain Challenge Solver")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Scenario to run (1, 2, or 3)",
    )
    parser.add_argument(
        "--player-id",
        type=str,
        default=DEFAULT_PLAYER_ID,
        help="Player ID for the game",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument(
        "--solver",
        type=str,
        choices=["greedy", "conservative", "lp", "dual"],
        default="lp",
        help="Solver strategy to use",
    )

    args = parser.parse_args()

    constraints = get_constraints(scenario=args.scenario)
    
    def create_solver():
        if args.solver == "lp":
            return LinearProgrammingSolver(
                attribute_frequencies=get_frequencies(scenario=args.scenario),
                correlation_matrix=get_corr(scenario=args.scenario),
                constraints=constraints,
                distribution=get_distribution(scenario=args.scenario),
            )
        elif args.solver == "dual":
            return DualThresholdSolver(
                attribute_frequencies=get_frequencies(scenario=args.scenario),
                correlation_matrix=get_corr(scenario=args.scenario),
                constraints=constraints,
                distribution=get_distribution(scenario=args.scenario),
            )
        else:
            raise ValueError(args.solver)

    if args.trials == 1:
        # Single trial - run directly
        solver = create_solver()
        result = solver.play_game(args.scenario, args.player_id, verbose=False)
        print(f"Result: {result} rejections")
    else:
        # Multiple trials - use ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(args.trials):
                solver = create_solver()
                future = executor.submit(run_single_trial, solver, args.scenario, args.player_id)
                futures.append(future)
            
            for future in tqdm(futures, desc="Running trials"):
                result = future.result()
                results.append(result)
        
        # Print summary statistics
        valid_results = [r for r in results if r != float('-inf')]
        if valid_results:
            avg_rejections = sum(valid_results) / len(valid_results)
            min_rejections = min(valid_results)
            max_rejections = max(valid_results)
            success_rate = len(valid_results) / len(results)
            
            print(f"Trials completed: {len(results)}")
            print(f"Success rate: {success_rate:.1%} ({len(valid_results)}/{len(results)})")
            print(f"Average rejections: {avg_rejections:.1f}")
            print(f"Min rejections: {min_rejections}")
            print(f"Max rejections: {max_rejections}")
        else:
            print(f"All {len(results)} trials failed")


if __name__ == "__main__":
    main()
