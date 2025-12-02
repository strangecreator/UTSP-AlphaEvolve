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


# SOLUTIONS_DIRECTORY = str(BASE_DIR / "temp/solutions")
SOLUTIONS_DIRECTORY = "/workspace/dataspace/alpha_evolve/UTSP-AlphaEvolve/temp/solutions"
N = 1000
INSTANCES_COUNT = 48


def parse_code_block(text: str, block_header: str) -> str | None:
    text_splitted = text.split(block_header, maxsplit=1)

    if len(text_splitted) < 2: return None

    block_code = text_splitted[1].strip()

    if not block_code.startswith("@@@\n"): return None
    
    block_code = block_code.removeprefix("@@@\n").strip()

    if "\n@@@" not in block_code: return None

    return block_code.split("\n@@@", maxsplit=1)[0].strip()


def retrieve_text_changes(text: str) -> str:
    code_block = parse_code_block(text, "* changes_description.txt *:")

    if code_block is None: return "Could not parse changes description (maybe it does not exist)."
    return code_block


def retrieve_text_changes_safe(text: str | None) -> str | None:
    if text is None:  return None
    return retrieve_text_changes(text)


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


# def calc_combined_score(
#     n: int,
#     distances: np.ndarray,
#     time_elapsed: list[float | int | None],
#     h: float = 1.0,
#     w: float = 1.0,
#     *,
#     alpha: float = 30.0,    # distance weight (bigger -> harsher on gap)
#     beta: float = 1.0,      # time weight (smaller than alpha)
#     g_cut: float = 0.15,    # hard cutoff on bad tours (set to None to disable)
#     eps_time: float = 1e-3  # floor for time scale
# ) -> float:
#     """
#     Distance-dominant combined score for TSP:
#       S_i = exp(-alpha * gap_i) * (1 + t_i / t_ref)^(-beta) * 1[g_i <= g_cut]
#       gap_i = max(0, distances[i] / L_star - 1)

#     Larger is better. Strongly prioritizes tour quality; time is a tie-breaker/regularizer.
#     """

#     if n < 3:
#         raise ValueError("n must be >= 3.")
#     if h <= 0 or w <= 0:
#         raise ValueError("h and w must be positive.")

#     dists = np.asarray(distances, dtype=np.float64).ravel()
#     m = dists.shape[0]
#     if len(time_elapsed) != m:
#         raise ValueError("len(time_elapsed) must match distances.shape[0].")

#     # Baseline optimal length via BHH
#     L_star = float(approximation_using_BHH_constant(n, h=h, w=w))
#     if not np.isfinite(L_star) or L_star <= 0:
#         raise ValueError("Invalid L_star computed from BHH approximation.")

#     # Relative gaps (clip negative)
#     gaps = np.maximum(0.0, dists / L_star - 1.0)

#     # Distance factor (dominant)
#     dist_factor = np.exp(-alpha * gaps)

#     # Optional hard cutoff for junk solutions
#     if g_cut is not None:
#         dist_factor = np.where(gaps <= g_cut, dist_factor, 0.0)

#     # Robust time scale t_ref = median of valid times (fallback to 1.0)
#     t_vals = []
#     for t in time_elapsed:
#         if t is not None and np.isfinite(t) and t > 0:
#             t_vals.append(float(t))
#     t_ref = float(np.median(t_vals)) if t_vals else 1.0
#     t_ref = max(t_ref, eps_time)

#     # Time factor (secondary). If time is invalid, treat as neutral (1.0),
#     # so distance truly dominates rather than zeroing the instance.
#     time_factor = np.empty(m, dtype=np.float64)
#     for i, t in enumerate(time_elapsed):
#         if t is None or not np.isfinite(t) or t <= 0:
#             time_factor[i] = 1.0  # neutral modifier; distance decides
#         else:
#             time_factor[i] = (1.0 + float(t) / t_ref) ** (-beta)

#     per_instance = dist_factor * time_factor
#     score = float(per_instance.mean()) if m > 0 else 0.0
#     return 0.0 if not np.isfinite(score) else score


def calc_combined_score(
    n: int,
    distances: np.ndarray,
    time_elapsed,  # Sequence[float | int | None]
    h: float = 1.0,
    w: float = 1.0,
    *,
    alpha: float = 30.0,          # distance weight (large -> very distance-dominant)
    g_cut: float | None = 0.15,   # hard cutoff on very bad tours (relative gap > g_cut)
    time_limit: float | None = 160.0,  # per-instance time budget in seconds
    time_weight: float = 0.10,    # how much time can perturb distance score (0 = ignore time)
    time_beta: float = 1.0,       # shape for time curve; >= 1 keeps it gentle
) -> float:
    """
    Distance-first combined score for TSP on [0,1] x [0,1] with n cities.

    For each instance i with tour length d_i and runtime t_i:

      1) BHH baseline:
           L_* = c_BHH * sqrt(n * h * w)

      2) Relative gap (clipped at zero):
           g_i = max(0, d_i / L_* - 1)

      3) Distance factor (dominant):
           D_i = exp(-alpha * g_i)                 if g_i <= g_cut
                 0                                  otherwise (junk tour)

      4) Normalized time (only if `time_limit` is not None):
           τ_i = clip(t_i, 0, time_limit) / time_limit        in [0,1]
           T_i = (1 - τ_i) ** time_beta                       in [0,1], higher is better

         Missing / invalid times are treated as worst case (τ_i = 1, T_i = 0).

      5) Time as a *small* modifier (tie-breaker):
           M_i = 1 - time_weight * (1 - T_i)

         So M_i ∈ [1 - time_weight, 1].
         - If time_weight = 0.10 then time can change D_i by at most ±10%.
         - Fast runs (T_i ~ 1) give M_i ~ 1.
         - Slow runs near the limit (T_i ~ 0) give M_i ~ 1 - time_weight.

      6) Per-instance score and final score:
           S_i = D_i * M_i
           Score = mean_i S_i

    Larger score is better. Distance dominates; time is only a gentle tie-breaker.
    """

    # BHH reference length
    L_star = approximation_using_BHH_constant(n, h=h, w=w)

    dists = np.asarray(distances, dtype=np.float64).ravel()
    m = dists.shape[0]

    if len(time_elapsed) != m:
        raise ValueError("len(time_elapsed) must match distances.shape[0].")

    # Start with everything as zero (worst case)
    per_instance = np.zeros(m, dtype=np.float64)

    # Valid distance mask
    valid_dist = np.isfinite(dists) & (dists > 0.0)
    if not np.any(valid_dist):
        return 0.0  # everything is garbage

    # Relative gaps for valid distances
    gaps = np.empty_like(dists)
    gaps[~valid_dist] = np.inf
    gaps[valid_dist] = np.maximum(0.0, dists[valid_dist] / L_star - 1.0)

    # Distance factor
    dist_factor = np.zeros_like(dists)
    good_gap_mask = valid_dist.copy()
    if g_cut is not None:
        good_gap_mask &= (gaps <= g_cut)

    dist_factor[good_gap_mask] = np.exp(-alpha * gaps[good_gap_mask])
    # for invalid or too-bad gaps dist_factor stays 0 -> per_instance stays 0

    # Time modifier
    if time_limit is None or time_weight <= 0.0:
        # Ignore time entirely, pure distance-based score
        per_instance = dist_factor
    else:
        t_max = max(float(time_limit), 1e-3)

        # τ_i = 1 (worst) by default
        tau = np.ones(m, dtype=np.float64)
        for i, t in enumerate(time_elapsed):
            if t is None:
                continue
            try:
                t_val = float(t)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(t_val) or t_val <= 0.0:
                continue
            # Clip at the time budget
            t_eff = min(t_val, t_max)
            tau[i] = t_eff / t_max

        # T_i in [0,1], higher is better (faster)
        T = (1.0 - tau) ** time_beta

        # M_i in [1 - time_weight, 1]
        time_modifier = 1.0 - time_weight * (1.0 - T)

        per_instance = dist_factor * time_modifier

    score = float(per_instance.mean())
    if not math.isfinite(score):
        return 0.0

    # Clamp to [0,1] for sanity; preserves ordering in practice
    return float(max(0.0, min(1.0, score)))


def evaluate(program_path: str) -> EvaluationResult:
    """Main stage evaluation with thorough testing on test dataset."""

    # parsing metadata (including lineage & parent changes)
    meta_path = (program_path + ".meta.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except Exception:
            metadata = {}
    
    # checking changes_description.txt
    parent_description = retrieve_text_changes_safe(metadata["parent_code"])
    current_description = retrieve_text_changes_safe(read_file(program_path))

    discard_program = False
    discard_reason = ''

    if current_description is None or current_description == '':
        discard_program = True
        discard_reason = "Missing or empty changes_description.txt"
    elif isinstance(parent_description, str) and parent_description.strip() == current_description:
        discard_program = True
        discard_reason = "changes_description.txt is identical to parent"

    if discard_program:
        return EvaluationResult(metrics={"combined_score": 0.0}, artifacts={
            "discard_program": True,
            "discard_reason": discard_reason,
            "changes_description": current_description,
            "parent_changes_description": parent_description,
        })

    # changes description looks fine, continuing evaluation

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
                heat_map_train_timeout=480.0,
                heat_map_inference_timeout=60.0,
                tsp_compilation_timeout=10.0,
                tsp_run_timeout=160.0,
                verbose=False,
                output_saver=output_saver,
            )
            remove_dir(solution_dir_path)
        except Exception as e:
            error_artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
            }
            build_artifacts_from_saver(error_artifacts, error_metrics, output_saver)            
            remove_dir(solution_dir_path)

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
    # print(evaluate(str(BASE_DIR / "evolve/initial_program.txt")))
    # print(evaluate(str(BASE_DIR / "best_programs/1000/23_51.txt")))

    print(calc_combined_score(1000, np.array([23.50, 23.54]), np.array([121, 125])))
    print(calc_combined_score(1000, np.array([23.49, 23.53]), np.array([90, 90])))