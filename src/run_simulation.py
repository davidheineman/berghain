import argparse
from typing import Dict, List, Tuple, Optional
from simulator import SimulatedGame
from dual_solver import DualThresholdSolver
from solvers import RejectAllSolver
from constants import get_constraints, get_corr, get_distribution, get_frequencies
from api import Constraint
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def run_single_simulation(constraints, solver, scenario):
    return run_simulation(constraints, solver, scenario)


def run_simulation(constraints: List[Constraint], solver, scenario: int = 1) -> Dict:
    game = SimulatedGame(constraints, solver, scenario)
    results = game.play_game()
    return results


def run_multiple_simulations(constraints, solver, num_runs=10, scenario=1, verbose: bool = True):
    all_results = []
    successful_runs = 0
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_simulation, constraints, solver, scenario) for i in range(num_runs)]
        for future in tqdm(futures, desc="Running simulations"):
            result = future.result()
            all_results.append(result)
            
            if result['success']:
                successful_runs += 1
    
    # Calculate averages
    successful_results = [r for r in all_results if r['success']]
    
    avg_results = {
        'success_rate': successful_runs / num_runs,
        'avg_admitted': sum(r['admitted'] for r in all_results) / num_runs,
        'avg_rejected': sum(r['rejected'] for r in all_results) / num_runs,
        'avg_rejected_successful': sum(r['rejected'] for r in successful_results) / len(successful_results) if successful_results else None,
        'min_rejected_successful': min(r['rejected'] for r in successful_results) if successful_results else None,
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
    if verbose:
        print(f"Success Rate: {successful_runs}/{num_runs} ({avg_results['success_rate']*100:.1f}%)")
        print(f"Average Admitted: {avg_results['avg_admitted']:.0f}/1000")
        print(f"Average Rejected: {avg_results['avg_rejected']:.0f}")
        if successful_results:
            print(f"Average Rejected (Successful Runs): {avg_results['avg_rejected_successful']:.0f}")
            print(f"Minimum Rejected (Successful Runs): {avg_results['min_rejected_successful']:.0f}")
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

def _sweep_dual_hyperparameters(
    constraints,
    scenario: int,
    runs: int,
    z0_list: List[float],
    z1_list: List[float],
    eta0_list: List[float],
    lambda_list: List[float],
    endgame_list: List[int],
    rarity_list: List[bool],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    best_config: Optional[Dict] = None
    best_results: Optional[Dict] = None

    total_combos = (
        max(1, len(z0_list))
        * max(1, len(z1_list))
        * max(1, len(eta0_list))
        * max(1, len(lambda_list))
        * max(1, len(endgame_list))
        * max(1, len(rarity_list))
    )
    combo_index = 0

    for z0 in (z0_list or [2.5]):
        for z1 in (z1_list or [1.0]):
            for eta0 in (eta0_list or [0.8]):
                for lambda_max in (lambda_list or [8.0]):
                    for endgame_R in (endgame_list or [50]):
                        for use_rarity in (rarity_list or [False]):
                            combo_index += 1
                            print(
                                f"\n=== Dual Sweep {combo_index}/{total_combos}: z0={z0}, z1={z1}, eta0={eta0}, lambda_max={lambda_max}, endgame_R={endgame_R}, use_rarity={use_rarity} ==="
                            )
                            solver = DualThresholdSolver(
                                attribute_frequencies=get_frequencies(scenario=scenario),
                                correlation_matrix=get_corr(scenario=scenario),
                                distribution=get_distribution(scenario=scenario),
                                constraints=constraints,
                                z0=z0,
                                z1=z1,
                                eta0=eta0,
                                lambda_max=lambda_max,
                                use_rarity=use_rarity,
                                endgame_R=endgame_R,
                            )
                            solver.initialize_policy(constraints)

                            results = run_multiple_simulations(
                                constraints,
                                solver,
                                num_runs=runs,
                                scenario=scenario,
                                verbose=False,
                            )

                            min_rej = results.get("min_rejected_successful")
                            if min_rej is None:
                                print(
                                    "No successful runs for this configuration; skipping from best selection."
                                )
                                continue

                            if best_results is None or min_rej < best_results.get(
                                "min_rejected_successful", float("inf")
                            ):
                                best_results = results
                                best_config = {
                                    "z0": z0,
                                    "z1": z1,
                                    "eta0": eta0,
                                    "lambda_max": lambda_max,
                                    "endgame_R": endgame_R,
                                    "use_rarity": use_rarity,
                                }

                            print(f"  -> Min Rejected (Successful): {min_rej}")

    return best_config, best_results


def main():
    parser = argparse.ArgumentParser(description='Run Berghain game simulation')
    parser.add_argument('--solver', choices=['lp', 'dual', 'rejectall'], default='lp', help='Solver to use')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3], default=1, help='Scenario to run (1, 2, or 3)')
    # DualThresholdSolver hyperparameters
    parser.add_argument('--dual_z0', type=float, default=2.5, help='DualThresholdSolver z0')
    parser.add_argument('--dual_z1', type=float, default=1.0, help='DualThresholdSolver z1')
    parser.add_argument('--dual_eta0', type=float, default=0.8, help='DualThresholdSolver initial step size eta0')
    parser.add_argument('--dual_lambda_max', type=float, default=8.0, help='DualThresholdSolver lambda_max')
    parser.add_argument('--dual_use_rarity', action='store_true', help='Enable rarity weighting in DualThresholdSolver')
    parser.add_argument('--dual_endgame_R', type=int, default=50, help='DualThresholdSolver endgame threshold R')
    # DualThresholdSolver sweep options
    parser.add_argument('--dual_sweep', action='store_true', help='Sweep DualThresholdSolver hyperparameters to minimize Min Rejected (Successful Runs)')
    parser.add_argument('--dual_z0_list', type=float, nargs='+', default=[], help='Values to sweep for dual z0')
    parser.add_argument('--dual_z1_list', type=float, nargs='+', default=[], help='Values to sweep for dual z1')
    parser.add_argument('--dual_eta0_list', type=float, nargs='+', default=[], help='Values to sweep for dual eta0')
    parser.add_argument('--dual_lambda_max_list', type=float, nargs='+', default=[], help='Values to sweep for dual lambda_max')
    parser.add_argument('--dual_endgame_R_list', type=int, nargs='+', default=[], help='Values to sweep for dual endgame_R')
    parser.add_argument('--dual_use_rarity_opts', type=int, nargs='+', choices=[0, 1], default=[], help='Values to sweep for dual use_rarity (0 or 1)')
    
    args = parser.parse_args()

    constraints = get_constraints(scenario=args.scenario)
    
    if args.solver == 'dual':
        # Sweep mode for DualThresholdSolver
        if args.dual_sweep:
            z0_list = args.dual_z0_list or [args.dual_z0]
            z1_list = args.dual_z1_list or [args.dual_z1]
            eta0_list = args.dual_eta0_list or [args.dual_eta0]
            lambda_list = args.dual_lambda_max_list or [args.dual_lambda_max]
            endgame_list = args.dual_endgame_R_list or [args.dual_endgame_R]
            # If opts provided use them; else derive from single flag
            if args.dual_use_rarity_opts:
                rarity_list = [bool(v) for v in args.dual_use_rarity_opts]
            else:
                rarity_list = [bool(args.dual_use_rarity)]

            best_config, best_results = _sweep_dual_hyperparameters(
                constraints=constraints,
                scenario=args.scenario,
                runs=args.runs,
                z0_list=z0_list,
                z1_list=z1_list,
                eta0_list=eta0_list,
                lambda_list=lambda_list,
                endgame_list=endgame_list,
                rarity_list=rarity_list,
            )

            if best_config is None:
                print("No successful configuration found across the dual sweep.")
                return {
                    'success_rate': 0.0,
                    'avg_admitted': None,
                    'avg_rejected': None,
                    'avg_rejected_successful': None,
                    'min_rejected_successful': None,
                    'avg_constraint_satisfaction': {},
                }

            print("\n=== Best Dual Configuration (by Minimum Rejected on Successful Runs) ===")
            print(
                f"z0={best_config['z0']}, z1={best_config['z1']}, eta0={best_config['eta0']}, "
                f"lambda_max={best_config['lambda_max']}, endgame_R={best_config['endgame_R']}, use_rarity={best_config['use_rarity']}"
            )
            print("Results:")
            print(f"  Success Rate: {best_results['success_rate']*100:.1f}%")
            print(f"  Average Admitted: {best_results['avg_admitted']:.0f}/1000")
            print(f"  Average Rejected: {best_results['avg_rejected']:.0f}")
            if best_results.get('avg_rejected_successful') is not None:
                print(f"  Average Rejected (Successful Runs): {best_results['avg_rejected_successful']:.0f}")
                print(f"  Minimum Rejected (Successful Runs): {best_results['min_rejected_successful']:.0f}")
            return best_results

        solver = DualThresholdSolver(
            distribution=get_distribution(scenario=args.scenario),
            attribute_frequencies=get_frequencies(scenario=args.scenario),
            correlation_matrix=get_corr(scenario=args.scenario),
            constraints=constraints,
            z0=args.dual_z0,
            z1=args.dual_z1,
            eta0=args.dual_eta0,
            lambda_max=args.dual_lambda_max,
            use_rarity=args.dual_use_rarity,
            endgame_R=args.dual_endgame_R,
        )
    elif args.solver == 'rejectall':
        solver = RejectAllSolver()
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    return run_multiple_simulations(constraints, solver, args.runs, args.scenario)


if __name__ == "__main__":
    main()
