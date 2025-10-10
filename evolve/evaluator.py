# standart imports
import os
import sys
import json
import uuid
import pathlib
import traceback
import numpy as np
from datetime import datetime

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

# openevolve
from openevolve.evaluation_result import EvaluationResult

# other imports
from utils import *
from runner import *
from load_data import *
from code_to_query import *


SOLUTIONS_DIRECTORY = str(BASE_DIR / "temp/solutions")
N = 500
INSTANCES_COUNT = 32


def calc_average_elapsed_time(time_elapsed: list[float | int | None]) -> float | None:
    """
    Calculate the average of non-None times.
    If more than 5% of the elements are None, return None.
    """

    if not time_elapsed:
        return None

    n_total = len(time_elapsed)
    n_none = sum(t is None for t in time_elapsed)

    # invalid if more than 5% are None
    if n_none / n_total > 0.05:
        return None

    valid_values = [t for t in time_elapsed if t is not None]
    if not valid_values:
        return None

    return float(sum(valid_values) / len(valid_values))


def build_artifacts_from_saver(artifacts: dict, metrics: dict, output_saver: dict) -> dict:
    if "heat_map_train_data" in output_saver:
        artifacts["heat_map_train_stdout"] = output_saver["heat_map_train_data"]["stdout"]
        artifacts["heat_map_train_stderr"] = output_saver["heat_map_train_data"]["stderr"]

        artifacts["heat_map_train_time_elapsed"] = output_saver["heat_map_train_data"]["time_elapsed"]
        metrics["heat_map_train_time_elapsed"] = output_saver["heat_map_train_data"]["time_elapsed"]
    
    if "heat_map_inference_data" in output_saver:
        artifacts["heat_map_inference_first_test_sample_stdout"] = output_saver["heat_map_inference_data"]["instance_stdout"]
        artifacts["heat_map_inference_first_test_sample_stderr"] = output_saver["heat_map_inference_data"]["instance_stderr"]

        average_heat_map_inference_time_elapsed = calc_average_elapsed_time(output_saver["heat_map_inference_data"]["time_elapsed"])
        artifacts["average_heat_map_inference_time_elapsed"] = average_heat_map_inference_time_elapsed
        metrics["average_heat_map_inference_time_elapsed"] = average_heat_map_inference_time_elapsed
    
    if "tsp_run_data" in output_saver:
        artifacts["tsp_run_first_test_sample_stdout"] = output_saver["tsp_run_data"]["instance_stdout"]
        artifacts["tsp_run_first_test_sample_stderr"] = output_saver["tsp_run_data"]["instance_stderr"]

        average_tsp_run_time_elapsed = calc_average_elapsed_time(output_saver["tsp_run_data"]["time_elapsed"])
        artifacts["average_tsp_run_time_elapsed"] = average_tsp_run_time_elapsed
        metrics["average_tsp_run_time_elapsed"] = average_tsp_run_time_elapsed
    
    return artifacts


def calc_combined_score(
    n: int,
    distances: np.ndarray,
    time_elapsed: list[float | int | None],
    h: float = 1.0,
    w: float = 1.0,
    *,
    alpha: float = 30.0,    # distance weight (bigger -> harsher on gap)
    beta: float = 1.0,      # time weight (smaller than alpha)
    g_cut: float = 0.15,    # hard cutoff on bad tours (set to None to disable)
    eps_time: float = 1e-3  # floor for time scale
) -> float:
    """
    Distance-dominant combined score for TSP:
      S_i = exp(-alpha * gap_i) * (1 + t_i / t_ref)^(-beta) * 1[g_i <= g_cut]
      gap_i = max(0, distances[i] / L_star - 1)

    Larger is better. Strongly prioritizes tour quality; time is a tie-breaker/regularizer.
    """

    if n < 3:
        raise ValueError("n must be >= 3.")
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive.")

    dists = np.asarray(distances, dtype=np.float64).ravel()
    m = dists.shape[0]
    if len(time_elapsed) != m:
        raise ValueError("len(time_elapsed) must match distances.shape[0].")

    # Baseline optimal length via BHH
    L_star = float(approximation_using_BHH_constant(n, h=h, w=w))
    if not np.isfinite(L_star) or L_star <= 0:
        raise ValueError("Invalid L_star computed from BHH approximation.")

    # Relative gaps (clip negative)
    gaps = np.maximum(0.0, dists / L_star - 1.0)

    # Distance factor (dominant)
    dist_factor = np.exp(-alpha * gaps)

    # Optional hard cutoff for junk solutions
    if g_cut is not None:
        dist_factor = np.where(gaps <= g_cut, dist_factor, 0.0)

    # Robust time scale t_ref = median of valid times (fallback to 1.0)
    t_vals = []
    for t in time_elapsed:
        if t is not None and np.isfinite(t) and t > 0:
            t_vals.append(float(t))
    t_ref = float(np.median(t_vals)) if t_vals else 1.0
    t_ref = max(t_ref, eps_time)

    # Time factor (secondary). If time is invalid, treat as neutral (1.0),
    # so distance truly dominates rather than zeroing the instance.
    time_factor = np.empty(m, dtype=np.float64)
    for i, t in enumerate(time_elapsed):
        if t is None or not np.isfinite(t) or t <= 0:
            time_factor[i] = 1.0  # neutral modifier; distance decides
        else:
            time_factor[i] = (1.0 + float(t) / t_ref) ** (-beta)

    per_instance = dist_factor * time_factor
    score = float(per_instance.mean()) if m > 0 else 0.0
    return 0.0 if not np.isfinite(score) else score


def evaluate(program_path: str) -> dict:
    """Main stage evaluation with thorough testing on test dataset."""

    # parsing metadata (which includes lineage)
    meta_path, metadata = (program_path + ".meta.json"), {}

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
    
    # creating a new solution directory id
    solution_dir_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '-' + str(uuid.uuid4())
    solution_dir_path = f"{SOLUTIONS_DIRECTORY}/{solution_dir_id}"

    # parsing solutions code and generating all files
    save_parsed_output_code(parse_output_code(read_file(program_path)), solution_dir_path)

    error_metrics = {
        "heat_map_train_time_elapsed": 0.0,
        "average_heat_map_inference_time_elapsed": 0.0,
        "average_tsp_run_time_elapsed": 0.0,
        "average_path_length": 0.0,
        "path_length_variance": 0.0,
        "combined_score": 0.0,
    }

    output_saver = {}

    # testing pipeline
    try:
        # loading cities
        cities = load_points(N, split="test", instances_count=INSTANCES_COUNT)

        # running
        try:
            run_data = run(
                solution_dir_path,
                cities,
                heat_map_train_timeout=360.0,
                heat_map_inference_timeout=60.0,
                tsp_compilation_timeout=10.0,
                tsp_run_timeout=120.0,
                verbose=False,
                output_saver=output_saver,
            )
        except Exception as e:
            error_artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
            build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)
            
            return EvaluationResult(
                metrics=error_metrics | {"error": str(e)},
                artifacts=error_artifacts
            )

        # checking solutions
        try:
            total_distances = calc_total_cycle_distance(cities, run_data["solutions"])
        except Exception as e:
            error_artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
            build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)
            return EvaluationResult(
                metrics=error_metrics | {"error": str(e)},
                artifacts=error_artifacts
            )

        # final metrics and artifacts
        artifacts, metrics = {}, {}
        build_artifacts_from_saver(artifacts, metrics, output_saver)

        metrics["average_path_length"] = float(total_distances.mean())
        metrics["path_length_variance"] = float(np.var(total_distances))
        metrics["combined_score"] = calc_combined_score(cities.shape[1], total_distances, output_saver["tsp_run_data"]["time_elapsed"], h=1.0, w=1.0)

        artifacts["average_path_length"] = metrics["average_path_length"]
        artifacts["path_length_variance"] = metrics["path_length_variance"]

        # returning results
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}.")
        print(traceback.format_exc())

        # create error artifacts
        error_artifacts, error_metrics = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code."
        }, {}
        build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)

        return EvaluationResult(
            metrics=error_metrics | {"error": str(e)},
            artifacts=error_artifacts
        )


if __name__ == "__main__":
    pass