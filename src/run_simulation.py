import argparse
from simulator import run_simulation
from lp_solver import LinearProgrammingSolver
from api import Constraint


def run_multiple_simulations(constraints, solver, num_runs=10):
    """Run multiple simulations and return averaged results."""
    print(f"Running {num_runs} simulations...")
    
    all_results = []
    successful_runs = 0
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        result = run_simulation(constraints, solver)
        all_results.append(result)
        
        if result['success']:
            successful_runs += 1
            print(f"✅ Run {i+1}: SUCCESS")
        else:
            print(f"❌ Run {i+1}: FAILED")
    
    # Calculate averages
    avg_results = {
        'success_rate': successful_runs / num_runs,
        'avg_admitted': sum(r['admitted'] for r in all_results) / num_runs,
        'avg_rejected': sum(r['rejected'] for r in all_results) / num_runs,
        'avg_constraint_satisfaction': {}
    }
    
    # Average constraint satisfaction
    for constraint in constraints:
        attr = constraint.attribute
        avg_satisfied = sum(r['constraint_status'][attr]['current'] for r in all_results) / num_runs
        avg_results['avg_constraint_satisfaction'][attr] = {
            'satisfied': avg_satisfied,
            'required': constraint.min_count,
            'percentage': avg_satisfied / constraint.min_count * 100
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-RUN SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Success Rate: {successful_runs}/{num_runs} ({avg_results['success_rate']*100:.1f}%)")
    print(f"Average Admitted: {avg_results['avg_admitted']:.0f}/1000")
    print(f"Average Rejected: {avg_results['avg_rejected']:.0f}")
    print("\nAverage Constraint Satisfaction:")
    for attr, stats in avg_results['avg_constraint_satisfaction'].items():
        status = "✓" if stats['satisfied'] >= stats['required'] else "✗"
        print(f"  {attr}: {stats['satisfied']:.0f}/{stats['required']} ({stats['percentage']:.1f}%) {status}")
    
    return avg_results

def main():
    parser = argparse.ArgumentParser(description='Run Berghain game simulation')
    parser.add_argument('--solver', choices=['lp'], default='lp', help='Solver to use')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--single', action='store_true', help='Run single simulation instead of multiple')
    
    args = parser.parse_args()
    
    constraints = [
        Constraint(attribute="techno_lover", min_count=650),
        Constraint(attribute="well_connected", min_count=450),
        Constraint(attribute="creative", min_count=300),
        Constraint(attribute="berlin_local", min_count=750)
    ]
    
    solver = LinearProgrammingSolver()
    
    # Set up statistics based on the provided distribution
    solver.attribute_frequencies = {
        'techno_lover': 0.6265000000000001,
        'well_connected': 0.4700000000000001,
        'creative': 0.06227,
        'berlin_local': 0.398
    }
    
    solver.correlation_matrix = {
        'techno_lover': {
            'techno_lover': 1,
            'well_connected': -0.4696169332674324,
            'creative': 0.09463317039891586,
            'berlin_local': -0.6549403815606182
        },
        'well_connected': {
            'techno_lover': -0.4696169332674324,
            'well_connected': 1,
            'creative': 0.14197259140471485,
            'berlin_local': 0.5724067808436452
        },
        'creative': {
            'techno_lover': 0.09463317039891586,
            'well_connected': 0.14197259140471485,
            'creative': 1,
            'berlin_local': 0.14446459505650772
        },
        'berlin_local': {
            'techno_lover': -0.6549403815606182,
            'well_connected': 0.5724067808436452,
            'creative': 0.14446459505650772,
            'berlin_local': 1
        }
    }
    
    # Initialize the LP policy
    if hasattr(solver, 'initialize_policy'):
        solver.initialize_policy(constraints)
    
    # Run simulation
    print(f"Running simulation with {args.solver} solver...")
    print("Game will continue until exactly 1000 people are admitted.")
    print(f"Constraints: {[f'{c.attribute}:{c.min_count}' for c in constraints]}")
    print()

    return run_multiple_simulations(constraints, solver, args.runs)


if __name__ == "__main__":
    main()
