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


def change_config_file(config_file_path: str, cities_number: int, input_path: str, output_path: str) -> None:
    config = json.loads(read_file(config_file_path))

    config["cities_number"] = cities_number
    config["input_path"] = input_path
    config["output_path"] = output_path

    with open(config_file_path, 'w') as file:
        file.write(json.dumps(config, indent=4, ensure_ascii=False))


def parse_output_file(output_file_path: str) -> np.ndarray:
    with open(output_file_path, 'w') as file:
        return np.array(list(map(int, file.read().strip().split())))


def run(
    dir_path: str,
    cities: np.ndarray,
    input_path: str,
    output_path: str,
    heat_map_train_timeout: 360.0,
    heat_map_inference_timeout: float = 60.0,
    tsp_compilation_timeout: float = 10.0,
    tsp_run_timeout: float = 60.0
) -> dict:
    # train
    heat_map_train_io = {"stdin": None, "stdout": None, "stderr": None}
    heat_map_train_time_elapsed = run_python_heat_map_train(f"{dir_path}/heat_map_train.py", heat_map_train_io, timeout=heat_map_train_timeout)

    # inference
    heat_map_results = run_heat_maps_parallel(f"{dir_path}/heat_map_inference.py", cities, f"{dir_path}/heat_map_results", heat_map_inference_timeout, max_workers=None)  # in format [(index, npy_path, time_elapsed), ...]

    # config changing
    change_config_file(f"{dir_path}/config.json", cities.shape[0], input_path, output_path)

    # compilation of TSP.cpp
    compile_tsp_executable(dir_path, timeout=tsp_compilation_timeout)
    
    # building input files
    for i in range(cities.shape[0]):
        format_input_file(f"{dir_path}/input_files/instance_{i:05d}.txt", cities[i], np.load(heat_map_results[i][1]))

    # running compiled binary
    ...

    # parsing output
    solution = parse_output_file(output_path)

    return {
        "solution": solution
    }


if __name__ == "__main__":
    run(
        str(BASE_DIR / "UTSP"),
        load_points(200, "test"),
        "/Users/dark-creator/solomon/self/openevolve-usage/UTSP/UTSP-AlphaEvolve/UTSP/sample_input.txt",
        "/Users/dark-creator/solomon/self/openevolve-usage/UTSP/UTSP-AlphaEvolve/UTSP/sample_output.txt",
        heat_map_train_timeout=360.0,
        heat_map_inference_timeout=60.0,
        tsp_compilation_timeout=10.0,
        tsp_run_timeout=60.0,
    )