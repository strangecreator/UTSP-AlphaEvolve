import re
import sys
import pathlib
import asyncio
import truststore

truststore.inject_into_ssl()
BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

# openevolve & related imports
from openevolve import OpenEvolve

# other imports
from utils import *
from code_to_query import *


def latest_checkpoint(dir_path: pathlib.Path) -> str | None:
    pairs: list[tuple[int, pathlib.Path]] = []

    for p in dir_path.glob("checkpoint_*"):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"checkpoint_(\d+)", p.name)  # digits at the end only
        if m:
            pairs.append((int(m.group(1)), p))

    if not pairs:
        return None

    # picking the largest numeric suffix, tie-break by mod time just in case
    _, path = max(pairs, key=lambda t: (t[0], t[1].stat().st_mtime))
    return str(path)


async def main(evolve, checkpoint_path: str | None):
    best_program = await evolve.run(checkpoint_path=checkpoint_path)

    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    DIR_PATH = str(BASE_DIR / "UTSP")

    # building an initial program
    # with open(str(BASE_DIR / "evolve/initial_program.txt"), 'w') as file:
    #     file.write(format_query_code(DIR_PATH))

    if len(sys.argv) > 1:
        initial_program_path = sys.argv[1]
    else:
        initial_program_path = str(BASE_DIR / "evolve/initial_program.txt")

    print(f"Initial program path: `{initial_program_path}`.")
    
    create_dir("/workspace/dataspace/alpha_evolve/UTSP-AlphaEvolve/temp/solutions")

    # system initialization
    evolve = OpenEvolve(
        initial_program_path=initial_program_path,
        evaluation_file=str(BASE_DIR / "evolve/evaluator.py"),
        config_path=str(BASE_DIR / "evolve/config.yaml"),
    )

    # latest checkpoint
    checkpoint_path = latest_checkpoint(BASE_DIR / "best_programs/1000/openevolve_output/checkpoints")
    print(f"Using checkpoint: '{checkpoint_path}'.")

    # running evolution
    asyncio.run(main(evolve, checkpoint_path))