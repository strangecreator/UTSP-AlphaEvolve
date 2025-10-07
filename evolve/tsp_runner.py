import os
import sys
import time
import pathlib
import platform
import subprocess
import typing as tp

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

from utils import *


def compile_tsp_executable(
    dir_path: str,
    timeout: float = 120.0,
    cxx: str = "g++",
    extra_flags: tp.Optional[tp.Sequence[str]] = None,
) -> pathlib.Path:
    """
    Compile TSP.cpp in `dir_path` with includes from `dir_path/include`
    and output binary to `dir_path/bin/runner`.

    Raises:
      FileNotFoundError: if sources/dirs missing or compiler not found.
      TimeoutError: if compilation exceeds `timeout_sec`.
      RuntimeError: if compiler returns non-zero exit code.
    Returns:
      Path to the produced runner binary.
    """

    source_path = f"{dir_path}/TSP.cpp"
    include_path = f"{dir_path}/include"
    output_dir_path = f"{dir_path}/bin"
    output_bin_path = f"{output_dir_path}/runner"

    if not is_file_exist(source_path):
        raise FileNotFoundError(f"Missing source file: \"{source_path}\".")
    if not is_file_exist(include_path):
        raise FileNotFoundError(f"Missing include directory: \"{include_path}\".")

    create_dir(output_dir_path)

    # base flags (shell-safe as a list)
    cmd = [
        cxx,
        "-std=gnu++17",
        "-O3",
        "-DNDEBUG",
        "-march=native",
        "-funroll-loops",
        "-ffast-math",
        "-I", "include",
        str(pathlib.Path(source_path).name),  # compile from cwd=`dir_path`
        "-o", output_bin_path,
    ]

    # linker/libs
    sysname = platform.system().lower()

    if sysname == "linux":
        cmd += ["-lpthread", "-lm", "-ldl"]
    elif sysname == "darwin":  # macOS has no -ldl
        cmd += ["-lpthread", "-lm"]
    else:
        # MinGW/Windows: keep it minimal, users can extend via `extra_flags`
        pass

    if extra_flags:
        cmd += list(extra_flags)

    # popen config (no shell, capture output)
    popen_kwargs = dict(
        cwd=str(dir_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    start_time = time.monotonic()

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError:
        # compiler not found
        raise

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # kill compiler process/group on timeout
        try:
            if os.name == "posix":
                os.killpg(process.pid, 9)  # SIGKILL
            else:
                proc.kill()
        except Exception as kill_err:
            # failed to kill on timeout
            raise

        raise TimeoutError(f"Compilation timed out after {timeout:.2f}s.")

    time_elapsed = time.monotonic() - start_time

    if process.returncode != 0:
        raise RuntimeError(f"Compilation failed with exit code {process.returncode}.\nSee compile_cpp.log.\n{stderr}.")

    if not is_file_exist(output_bin_path):
        # extremely rare, but be explicit
        raise RuntimeError(f"Compilation reported success, but binary not found: \"{output_bin_path}\".")

    return output_bin_path


if __name__ == "__main__":
    DIR_PATH = str(BASE_DIR / "UTSP")

    print(compile_tsp_executable(DIR_PATH, timeout=10.0))