import os
import pathlib
import typing as tp
import concurrent.futures


BHH_CONSTANT_2D = 0.7120


def get_cpu_cores_number() -> int:
    return os.cpu_count()


def default_for_none(value: tp.Any, default: tp.Any) -> tp.Any:
    if value is None: return default
    return value


def create_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def touch_file(file_path: str) -> None:
    file_path = pathlib.Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)


def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()


def is_file_exist(file_path: str) -> bool:
    return pathlib.Path(file_path).exists()


def approximation_using_BHH_constant(n: int, h: float = 1.0, w: float = 1.0) -> float:
    return BHH_CONSTANT_2D * ((n * h * w) ** 0.5)


def run_with_timeout(func: tp.Callable, args=(), kwargs={}, timeout: float = 5.0):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run.
        args: Arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        timeout_seconds: Timeout in seconds.

    Returns:
        Result of the function or raises TimeoutError.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout} seconds.")


if __name__ == "__main__":
    print(approximation_using_BHH_constant(200))  # for a unit square