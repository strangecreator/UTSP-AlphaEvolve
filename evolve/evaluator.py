# standart imports
import os
import sys
import time
import json
import pathlib
import traceback
import numpy as np
import importlib.util
import concurrent.futures

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

# openevolve
from openevolve.evaluation_result import EvaluationResult

# other imports
from runner import *


def evaluate_stage1(program_path: str) -> dict:
    """Main stage evaluation with thorough testing on test dataset."""

    meta_path, metadata = (program_path + ".meta.json"), {}

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)

    try:
        return EvaluationResult(
            metrics={
                "heat_map_train_time_elapsed": reliability_score,
                "average_heat_map_inference_time_elapsed": average_heat_map_inference_time_elapsed,
                "average_tsp_run_time_elapsed": average_tsp_run_time_elapsed,

                "average_path_length": average_path_length,
                "path_length_variance": path_length_variance,

                "combined_score": combined_score,
            },
            artifacts=artifacts
        )
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}.")
        print(traceback.format_exc())

        # create error artifacts
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code."
        }

        return EvaluationResult(
            metrics={
                "value_score": 0.0,
                "distance_score": 0.0,
                "reliability_score": 0.0,
                "combined_score": 0.0,
                "error": str(e),
            },
            artifacts=error_artifacts
        )


if __name__ == "__main__":
    pass