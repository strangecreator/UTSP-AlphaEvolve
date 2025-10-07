import os
import sys
import time
import json
import psutil
import pathlib
import asyncio
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


def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _estimate_mem_gb_for_config(
    config_file_path: str,
    default_dtype_bytes: int = 4,
    alpha: float = 2.0,
    beta_gb: float = 0.5,
) -> float:
    """
    Heuristic: if config has a heat-map file path -> use its file size,
    else if it has N and dtype -> N^2 * sizeof(dtype),
    else return 0 to be replaced by global fallback.
    """
    
    n = json.loads(read_file(config_file_path))["cities_number"]
    bytes_ = n * n * default_dtype_bytes
    return alpha * (bytes_ / 1e9) + beta_gb


async def _run_one(
    index: int,
    runner_path: str,
    config_path: str,
    timeout: float | None,
    capture_stdout: bool,  # if to capture stdout, stderr
    env_overrides: dict[str, str] | None,
) -> tuple[int, int, float, str | None, str]:
    """
    Returns: (idx, returncode, elapsed, stdout_or_None, stderr)
    Raises TimeoutError on timeout. On cancel/error, ensures the child is killed.
    """

    # avoid CPU oversubscription from BLAS/OpenMP inside each process
    base_env = os.environ.copy()
    base_env.setdefault("OMP_NUM_THREADS", "1")
    base_env.setdefault("MKL_NUM_THREADS", "1")
    base_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    base_env.setdefault("NUMEXPR_NUM_THREADS", "1")
    base_env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if env_overrides:
        base_env.update(env_overrides)

    # Process group so we can kill all child threads/subprocesses
    creationflags = 0
    start_new_session = False
    if os.name == "posix":
        start_new_session = True
    else:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    stdout_target = asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL
    stderr_target = asyncio.subprocess.PIPE  # keep for diagnostics for *any* failure

    process = await asyncio.create_subprocess_exec(
        runner_path, config_path,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=stdout_target,
        stderr=stderr_target,
        start_new_session=start_new_session,
        creationflags=creationflags,
        env=base_env,
    )

    start_time = time.monotonic()

    try:
        if timeout is None or timeout <= 0:
            stdout, stderr = await process.communicate()
        else:
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                # kill hard on timeout
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)  # type: ignore[arg-type]
                    else:
                        process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                        await asyncio.sleep(0.2)
                        process.kill()
                finally:
                    raise TimeoutError(f"[{index}] timed out after {timeout}s: {config_path}")
        
        return_code = process.returncode

        time_elapsed = time.monotonic() - start_time

        out_s = stdout.decode("utf-8", "replace") if (stdout is not None and capture_stdout) else None
        err_s = stderr.decode("utf-8", "replace") if stderr is not None else ""

        if return_code != 0:
            # include some of stderr in the exception up the stack, caller will cancel others
            raise RuntimeError(f"[{index}] exit {return_code} for \"{config_path}\"\n stderr:\n{err_s[:4000]}")
        
        return index, return_code, time_elapsed, out_s, err_s

    except asyncio.CancelledError:
        # ensure the child is dead when the group cancels us
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)  # type: ignore[arg-type]
            else:
                process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                await asyncio.sleep(0.2)
                process.kill()
        finally:
            raise


def _decide_workers(
    config_paths: list[str],
    max_workers: int | None,
    mem_per_proc_gb: float | None = None,
    alpha: float = 2.0,
    beta_gb: float = 0.5,
    safety_margin_gb: float = 2.0,
) -> tuple[int, float]:
    """
    Returns (workers, chosen_mem_per_proc_gb).
    """

    cpu_bound = max(1, get_cpu_cores_number() or 1)

    if mem_per_proc_gb is None or mem_per_proc_gb <= 0:
        # estimate from configs, take worst-case
        ests = []

        for config_path in config_paths:
            ests.append(_estimate_mem_gb_for_config(config_path, alpha=alpha, beta_gb=beta_gb))

        mem_per_proc_gb = max([e for e in ests if e > 0] or [1.0])  # fallback 1GB

    avail = max(0.0, _available_ram_gb() - safety_margin_gb)
    mem_bound = max(1, int(avail // max(mem_per_proc_gb, 0.5)))

    if max_workers and max_workers > 0:
        return (max(1, min(cpu_bound, mem_bound, max_workers))), mem_per_proc_gb

    return (max(1, min(cpu_bound, mem_bound))), mem_per_proc_gb


def run_runner_parallel(
    runner_path: str,
    config_paths: tp.Sequence[str],
    timeout: float | None = None,                # in seconds
    max_workers: int | None = None,              # cap by CPU
    mem_per_proc_gb: float | None = None,        # if known; else auto-estimate
    alpha: float = 2.0,                          # multiplicative overhead on heat-map bytes
    beta_gb: float = 0.5,                        # additive overhead per proc
    safety_margin_gb: float = 2.0,               # leave headroom
    capture_index: int = 0,                      # which input index to capture stdout/stderr for
    env_overrides: dict[str, str] | None = None, # e.g., {"OMP_NUM_THREADS": "2"}
) -> dict:
    """
    Launches {runner} for each config.json (one arg). Cancels all on first failure.
    Returns:
      {
        "exit_codes": [int,...]  # 0 for success,
        "elapsed":    [float,...],
        "first_stdout": str,
        "first_stderr": str,
        "used_workers": int,
        "mem_per_proc_gb": float,
      }
    """

    used_workers, chosen_mem = _decide_workers(config_paths, max_workers, mem_per_proc_gb, alpha, beta_gb, safety_margin_gb)

    async def _driver() -> dict:
        sem = asyncio.Semaphore(used_workers)
        results: list[tuple[int, int, float, str | None, str] | None] = [None] * len(config_paths)
        first_out: str = ""
        first_err: str = ""

        async def _one(i: int):
            async with sem:
                capture = (i == capture_index)
                index, rc, instance_time_elapsed, out_s, err_s = await _run_one(i, runner_path, config_paths[i], timeout, capture, env_overrides)
                results[index] = (index, rc, instance_time_elapsed, out_s, err_s)

        # TaskGroup cancels the rest on first exception
        try:
            async with asyncio.TaskGroup() as tg:  # py311+
                for i in range(len(config_paths)):
                    tg.create_task(_one(i))
        except Exception:
            # propagate – siblings already cancelled & killed by _run_one
            raise

        # collect ordered outputs
        exit_codes, time_elapsed = [], []

        for rec in results:
            assert rec is not None, "missing result (internal error)"

            i, rc, t, out_s, err_s = rec
            exit_codes.append(rc)
            time_elapsed.append(t)

            if i == capture_index:
                first_out = out_s or ""
                first_err = err_s or ""

        return {
            "exit_codes": exit_codes,
            "time_elapsed": time_elapsed,
            "instance_stdout": first_out,
            "instance_stderr": first_err,
            "used_workers": used_workers,
            "mem_per_proc_gb": chosen_mem,
        }

    return asyncio.run(_driver())


if __name__ == "__main__":
    DIR_PATH = str(BASE_DIR / "UTSP")

    print(compile_tsp_executable(DIR_PATH, timeout=10.0))