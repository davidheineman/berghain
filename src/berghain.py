import argparse
from lp_solver import LinearProgrammingSolver
from constants import get_constraints, get_corr, get_distribution, get_frequencies
from dual_solver import DualThresholdSolver

DEFAULT_PLAYER_ID = "e44d1e27-daad-4003-bc7d-fbd97d992269"


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
    if args.solver == "lp":
        solver = LinearProgrammingSolver(
            attribute_frequencies=get_frequencies(scenario=args.scenario),
            correlation_matrix=get_corr(scenario=args.scenario),
            constraints=constraints,
            distribution=get_distribution(scenario=args.scenario),
        )
    elif args.solver == "dual":
        solver = DualThresholdSolver(
            attribute_frequencies=get_frequencies(scenario=args.scenario),
            correlation_matrix=get_corr(scenario=args.scenario),
            constraints=constraints,
            distribution=get_distribution(scenario=args.scenario),
        )
        solver.initialize_policy(constraints)
    else:
        raise ValueError(args.solver)
    result = solver.play_game(args.scenario, args.player_id, verbose=False)
    print(f"Result: {result} rejections")


if __name__ == "__main__":
    main()
