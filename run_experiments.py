"""Main entry point for running market making experiments."""

import sys
from src.experiments import run_all_experiments, run_experiment_1, run_experiment_2, run_experiment_3


def main():
    """Run market making experiments."""
    if len(sys.argv) > 1:
        exp_num = sys.argv[1]
        if exp_num == "1":
            run_experiment_1()
        elif exp_num == "2":
            run_experiment_2()
        elif exp_num == "3":
            run_experiment_3()
        else:
            print(f"Unknown experiment: {exp_num}")
            print("Usage: python run_experiments.py [1|2|3]")
            print("  Run without arguments to execute all experiments")
    else:
        run_all_experiments()


if __name__ == "__main__":
    main()
