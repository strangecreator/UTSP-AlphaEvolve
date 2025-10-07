import os
import io
import sys
import time
import signal
import pathlib
import asyncio
import subprocess
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

from utils import *


def run_python_heat_map_train(
    program_path: str,
    heat_map_io: dict,
    timeout: float,
    python_executable: str = sys.executable,
) -> dict:
    """
    Runs python program as a subprocess, feeds cities via stdin, returns np.ndarray heat map.
    Raises TimeoutError on timeout; raises RuntimeError on non-zero exit.
    """

    cmd = [python_executable, program_path]
    heat_map_io["stdin"] = ''

    popen_kwargs = dict(
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    # ensure we can kill the whole process group on POSIX
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:  # Windows
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    time_start = time.monotonic()
    process = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]

    try:
        stdout, stderr = process.communicate('', timeout=timeout)
        heat_map_io["stdout"] = stdout
        heat_map_io["stderr"] = stderr
    except subprocess.TimeoutExpired:
        # robust kill (group on POSIX, process on Windows)
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception as kill_error:
            raise
        
        raise TimeoutError(f"Time exceeded {timeout} seconds.")

    time_elapsed = time.monotonic() - time_start

    if process.returncode != 0:
        raise RuntimeError(f"Program failed with exit code {process.returncode}.")

    return time_elapsed


def format_heat_map_stdin(cities: np.ndarray) -> str:
    return f"{cities.shape[0]}\n" + "\n".join(f"{x:.17g} {y:.17g}" for x, y in cities)


def parse_heat_map_stdout(stdout: str, n: int) -> np.ndarray:
    array = np.atleast_2d(np.loadtxt(io.StringIO(stdout)))

    if array.shape != (n, n):
        raise ValueError(f"Expected heat map of shape ({n}, {n}), but got {array.shape}.")

    return array


def _default_workers(n: tp.Optional[int]) -> int:
    if n and n > 0:
        return n
    
    cores_count = get_cpu_cores_number() or 1
    return cores_count


def run_python_heat_map_inference(
    program_path: str,
    heat_map_io: dict,
    cities: np.ndarray,
    timeout: float,
    python_executable: str = sys.executable,
) -> dict:
    """
    Runs python program as a subprocess, feeds cities via stdin, returns np.ndarray heat map.
    Raises TimeoutError on timeout; raises RuntimeError on non-zero exit.
    """

    cmd = [python_executable, program_path]
    stdin = format_heat_map_stdin(cities)
    heat_map_io["stdin"] = stdin

    popen_kwargs = dict(
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    # ensure we can kill the whole process group on POSIX
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:  # Windows
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    time_start = time.monotonic()
    process = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]

    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
        heat_map_io["stdout"] = stdout
        heat_map_io["stderr"] = stderr
    except subprocess.TimeoutExpired:
        # robust kill (group on POSIX, process on Windows)
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception as kill_error:
            raise
        
        raise TimeoutError(f"Time exceeded {timeout} seconds.")

    time_elapsed = time.monotonic() - time_start

    if process.returncode != 0:
        raise RuntimeError(f"Program failed with exit code {process.returncode}.")

    try:
        heat_map = parse_heat_map_stdout(stdout, cities.shape[0])
    except Exception as parse_error:
        raise

    return {
        "heat_map": heat_map,
        "time_elapsed": time_elapsed,
    }


async def _run_one_inference(
    index: int,
    program_path: str,
    python_executable: str,
    cities_i: np.ndarray,
    out_path: str,
    timeout: float,
) -> tuple[int, str, float]:
    """
    Launch one child, stream coords via stdin, child writes .npy to out_path.
    Returns (idx, out_path_str, elapsed_sec). Raises on error/timeout.
    Kills the whole process group on cancel/timeout.
    """

    stdin_bytes = format_heat_map_stdin(cities_i).encode("utf-8")

    # POSIX: kill process group. Windows: create new group and send CTRL_BREAK
    creationflags = 0
    start_new_session = False
    if os.name == "posix":
        start_new_session = True
    else:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    process = await asyncio.create_subprocess_exec(
        python_executable, program_path, "--out", out_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,  # expected to be tiny ("OK")
        stderr=asyncio.subprocess.PIPE,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )

    start_time = time.monotonic()

    try:
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin_bytes), timeout=timeout)
        except asyncio.TimeoutError:
            # hard kill on timeout
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)   # type: ignore[arg-type]
                else:
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    await asyncio.sleep(0.2)
                    process.kill()
            finally:
                raise TimeoutError(f"[{index}] timed out after {timeout}s.")
        
        return_code = process.returncode

        time_elapsed = time.monotonic() - start_time

        if return_code != 0:
            error = (stderr.decode("utf-8", "ignore") if stderr else '').strip()
            raise RuntimeError(f"[{index}] exited with {return_code} return code. stderr:\n{error[:4000]}")
        
        # quick shape check without loading into RAM
        array = np.load(out_path, mmap_mode='r')

        n = cities_i.shape[0]
        if array.shape != (n, n):
            raise ValueError(f"[{index}] bad shape {array.shape} in {out_path}, expected ({n}, {n})")
        
        return index, out_path, time_elapsed

    except asyncio.CancelledError:
        # cancellation path: ensure child is dead
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)  # type: ignore[arg-type]
            else:
                process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                await asyncio.sleep(0.2)
                process.kill()
        finally:
            raise


def run_heat_maps_parallel(
    program_path: str,
    cities: np.ndarray,  # shape: (B, n, 2)
    out_dir: str,
    timeout: float,  # in seconds
    max_workers: tp.Optional[int] = None,
    python_executable: str = sys.executable,
) -> list[str]:
    """
    Runs B instances in parallel (max_workers concurrently).
    On ANY failure: cancels the rest, kills children, re-raises the first error.
    On success: returns list of .npy paths in the SAME order as input.
    """

    create_dir(out_dir)

    B = cities.shape[0]
    assert cities.shape[2] == 2, "`cities` must be (B, n, 2)."

    out_paths = [f"{out_dir}/heat_map_{i:05d}.npy" for i in range(B)]

    async def _runner() -> list[str]:
        sem = asyncio.Semaphore(_default_workers(max_workers))
        results: list[tp.Optional[tuple[int, str, float]]] = [None] * B

        async def _guarded(i: int) -> None:
            async with sem:
                index, path, time_elapsed = await _run_one_inference(i, program_path, python_executable, cities[i], out_paths[i], timeout=timeout)
                results[index] = (index, path, time_elapsed)

        # TaskGroup auto-cancels siblings on first exception (Py3.11+)
        try:
            tg = getattr(asyncio, "TaskGroup", None)

            if tg is None:
                # fallback for <3.11
                tasks = [asyncio.create_task(_guarded(i)) for i in range(B)]
                await asyncio.gather(*tasks)
            else:
                async with asyncio.TaskGroup() as group:  # type: ignore[attr-defined]
                    for i in range(B):
                        group.create_task(_guarded(i))
        except Exception:
            # any error: best-effort cleanup of not-yet-written tmp files
            for out_path in out_paths:
                tmp = pathlib.Path(out_path + ".tmp")

                if tmp.exists():
                    try: tmp.unlink()
                    except Exception: pass

            raise

        # build ordered list
        ordered = []

        for r in results:
            assert r is not None, "Internal error: missing result."
            ordered.append(r)

        return ordered

    return asyncio.run(_runner())


if __name__ == "__main__":
    cities = np.random.uniform(0, 1, size=(10_000, 2))
    dir_path = str(BASE_DIR / "UTSP")