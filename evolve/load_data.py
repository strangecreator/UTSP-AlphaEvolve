import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

# torch & related imports
import numpy as np

# other imports
from utils import *


def generate_random_dataset(file_path: str, n: int, instances_number: int = 128) -> None:
    np.save(file_path, np.random.uniform(0, 1, size=(instances_number, n, 2)))


def _load_points(file_path: str) -> np.ndarray:
    return np.load(file_path)


def load_points(n: int, split: str = "train", instances_count: int = 128) -> np.ndarray:
    file_path = str(BASE_DIR / f"data/{split}/points/{n}.npy")

    if not is_file_exist(file_path):
        # generating a new dataset
        generate_random_dataset(file_path, n, instances_number=instances_count)

    return _load_points(file_path)[:instances_count]


def _load_solution(file_path: str) -> np.ndarray:
    return np.load(file_path)


def load_solution(n: int, split: str = "train") -> np.ndarray:
    file_path = str(BASE_DIR / f"data/{split}/solutions/{n}.npy")

    if not is_file_exist(file_path):
        raise ValueError("There is no saved solution for this `N`. File does not exist.")
    
    return _load_solution(file_path)


if __name__ == "__main__":
    points = load_points(10_000, split="test")
    print(points.shape)