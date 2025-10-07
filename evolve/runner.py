import sys
import json
import pathlib
import importlib.util

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

# other imports
from utils import *
from load_data import *
from tsp_runner import *
from heat_map_runner import *


def format_input_file(input_file_path: str, cities: np.ndarray, heat_map: np.ndarray) -> None:
    with open(input_file_path, 'w') as file:
        cities_str = '\n'.join(map(lambda x: ' '.join(map(str, x)), cities))
        heat_map_str = '\n'.join(map(lambda x: ' '.join(map(str, x)), heat_map))
        file.write(f"{cities.shape[0]}\n{cities_str}\n{heat_map_str}")


def change_config_file(input_config_file_path: str, output_config_file_path: str, cities_number: int, input_path: str, output_path: str) -> None:
    config = json.loads(read_file(input_config_file_path))

    config["cities_number"] = cities_number
    config["input_path"] = input_path
    config["output_path"] = output_path

    with open(output_config_file_path, 'w') as file:
        file.write(json.dumps(config, indent=4, ensure_ascii=False))


def parse_output_files(output_files_paths: list[str]) -> np.ndarray:
    results = []

    for output_file_path in output_files_paths:
        with open(output_file_path, 'w') as file:
            results.append( np.array(list(map(int, file.read().strip().split()))) )
    
    return results


def run(
    dir_path: str,
    cities: np.ndarray,
    heat_map_train_timeout: 360.0,
    heat_map_inference_timeout: float = 60.0,
    tsp_compilation_timeout: float = 10.0,
    tsp_run_timeout: float = 60.0,
    verbose: bool = False,
) -> dict:
    create_dir(f"{dir_path}/input_files")
    create_dir(f"{dir_path}/output_files")
    create_dir(f"{dir_path}/config_files")

    input_paths = [f"{dir_path}/input_files/instance_{i:05d}.txt" for i in range(cities.shape[0])]
    output_paths = [f"{dir_path}/output_files/instance_{i:05d}.txt" for i in range(cities.shape[0])]
    config_paths = [f"{dir_path}/config_files/instance_{i:05d}.txt" for i in range(cities.shape[0])]

    # train
    heat_map_train_io = {"stdin": None, "stdout": None, "stderr": None}
    heat_map_train_data = run_python_heat_map_train(f"{dir_path}/heat_map_train.py", heat_map_train_io, timeout=heat_map_train_timeout)
    if verbose: print("Heat map train complete.")

    # inference
    heat_map_inference_results = run_heat_maps_parallel(f"{dir_path}/heat_map_inference.py", cities, f"{dir_path}/heat_map_results", heat_map_inference_timeout, max_workers=None)  # in format [(index, npy_path, time_elapsed), ...]
    if verbose: print("Heat map inference complete.")

    # compilation of TSP.cpp
    compile_tsp_executable(dir_path, timeout=tsp_compilation_timeout)
    if verbose: print("Compilation complete.")

    # building input files
    for i in range(cities.shape[0]):
        format_input_file(input_paths[i], cities[i], np.load(heat_map_inference_results[i][1]))
    if verbose: print("Building input files complete.")

    # config changing
    for i in range(cities.shape[0]):
        change_config_file(f"{dir_path}/config.json", config_paths[i], cities.shape[0], input_paths[i], output_paths[i])
    if verbose: print("Building config files complete.")

    # running compiled binary
    tsp_run_data = run_runner_parallel(f"{dir_path}/bin/runner", config_paths, timeout=tsp_run_timeout)
    if verbose: print("TSP run complete.")

    # parsing output
    solutions = parse_output_files(output_paths)
    if verbose: print("Parsing output files complete.")

    return {
        "heat_map_train_data": heat_map_train_data,
        "heat_map_inference_data": heat_map_inference_results,
        "tsp_run_data": tsp_run_data,
        "solutions": solutions,
    }


if __name__ == "__main__":
    result = run(
        str(BASE_DIR / "UTSP"),
        load_points(200, "test"),
        "/Users/dark-creator/solomon/self/openevolve-usage/UTSP/UTSP-AlphaEvolve/UTSP/sample_input.txt",
        "/Users/dark-creator/solomon/self/openevolve-usage/UTSP/UTSP-AlphaEvolve/UTSP/sample_output.txt",
        heat_map_train_timeout=360.0,
        heat_map_inference_timeout=60.0,
        tsp_compilation_timeout=10.0,
        tsp_run_timeout=60.0,
        verbose=True,
    )

    print(result)