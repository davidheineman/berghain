import argparse
import time
from constants import get_constraints, get_corr, get_distribution, get_frequencies
from dual_solver import DualThresholdSolver
from solvers import RejectAllSolver




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
        choices=["greedy", "conservative", "lp", "dual", "rejectall"],
        default="lp",
        help="Solver strategy to use",
    )

    args = parser.parse_args()

    constraints = get_constraints(scenario=args.scenario)
    
    def create_solver():
        if args.solver == "dual":
            return DualThresholdSolver(
                attribute_frequencies=get_frequencies(scenario=args.scenario),
                correlation_matrix=get_corr(scenario=args.scenario),
                constraints=constraints,
                distribution=get_distribution(scenario=args.scenario),
            )
        elif args.solver == "rejectall":
            return RejectAllSolver()
        else:
            raise ValueError(args.solver)

    if args.trials == 1:
        # Single trial - run directly
        solver = create_solver()
        result = solver.play_game(args.scenario, args.player_id, verbose=True)
        print(f"Result: {result} rejections")
    else:
        # Multiple trials - keep running until interrupted
        all_results = []
        batch_num = 1
        
        try:
            while True:
                print(f"\n=== Batch {batch_num} ===")
                batch_results = []
                
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for i in range(args.trials):
                        solver = create_solver()
                        future = executor.submit(run_single_trial, solver, args.scenario, args.player_id)
                        futures.append(future)
                    
                    for future in tqdm(futures, desc=f"Running batch {batch_num}"):
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as e:
                            print(f"Trial failed with error: {e}")
                            batch_results.append(float('-inf'))
                
                all_results.extend(batch_results)
                
                # Print batch statistics
                valid_batch_results = [r for r in batch_results if r != float('-inf')]
                if valid_batch_results:
                    avg_rejections = sum(valid_batch_results) / len(valid_batch_results)
                    min_rejections = min(valid_batch_results)
                    max_rejections = max(valid_batch_results)
                    success_rate = len(valid_batch_results) / len(batch_results)
                    
                    print(f"Batch {batch_num} - Trials: {len(batch_results)}, Success: {success_rate:.1%}, Avg: {avg_rejections:.1f}, Min: {min_rejections}, Max: {max_rejections}")
                else:
                    print(f"Batch {batch_num} - All {len(batch_results)} trials failed")
                
                # Print cumulative statistics
                valid_all_results = [r for r in all_results if r != float('-inf')]
                if valid_all_results:
                    cumulative_avg = sum(valid_all_results) / len(valid_all_results)
                    cumulative_min = min(valid_all_results)
                    cumulative_max = max(valid_all_results)
                    cumulative_success_rate = len(valid_all_results) / len(all_results)
                    
                    print(f"Cumulative - Trials: {len(all_results)}, Success: {cumulative_success_rate:.1%}, Avg: {cumulative_avg:.1f}, Min: {cumulative_min}, Max: {cumulative_max}")
                
                batch_num += 1

                # Wait 5 minutes between batches with progress bar
                wait_time = 300
                for _ in tqdm(range(wait_time), desc="Waiting between batches", unit="s"):
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\nStopped after {len(all_results)} total trials across {batch_num - 1} batches")
            
            # Final summary
            valid_results = [r for r in all_results if r != float('-inf')]
            if valid_results:
                final_avg = sum(valid_results) / len(valid_results)
                final_min = min(valid_results)
                final_max = max(valid_results)
                final_success_rate = len(valid_results) / len(all_results)
                
                print(f"Final Summary:")
                print(f"  Total trials: {len(all_results)}")
                print(f"  Success rate: {final_success_rate:.1%} ({len(valid_results)}/{len(all_results)})")
                print(f"  Average rejections: {final_avg:.1f}")
                print(f"  Min rejections: {final_min}")
                print(f"  Max rejections: {final_max}")
            else:
                print(f"All {len(all_results)} trials failed")


if __name__ == "__main__":
    main()
