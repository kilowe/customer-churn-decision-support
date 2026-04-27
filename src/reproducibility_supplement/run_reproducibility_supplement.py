from __future__ import annotations

import sys
from pathlib import Path


# Ensure src/ is on sys.path so scripts can import existing project modules.
# This keeps imports explicit and avoids hidden environment assumptions.
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reproducibility_supplement.dss_outputs import generate_dss_outputs
from reproducibility_supplement.final_test_evaluation import evaluate_on_test_set
from reproducibility_supplement.threshold_optimization import run_threshold_optimization
from reproducibility_supplement.validation_pipeline import fit_models_and_export_probabilities


def run_full_supplement() -> None:
    """Run the reproducibility supplement in the documented sequence."""
    print("\nRunning reproducibility supplement pipeline...")

    print("\n[1/4] Running validation pipeline")
    fit_models_and_export_probabilities()

    print("\n[2/4] Running threshold optimization on validation set")
    run_threshold_optimization()

    print("\n[3/4] Running final test evaluation")
    evaluate_on_test_set()

    print("\n[4/4] Generating DSS outputs")
    generate_dss_outputs()

    print("\nReproducibility supplement completed successfully.")


if __name__ == "__main__":
    run_full_supplement()
