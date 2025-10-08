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


async def main(evolve, checkpoint_path: str | None):
    best_program = await evolve.run(checkpoint_path=checkpoint_path)

    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    DIR_PATH = str(BASE_DIR / "UTSP")

    # building an initial program
    with open(str(BASE_DIR / "evolve/initial_program.txt"), 'w') as file:
        file.write(format_query_code(DIR_PATH))
    
    create_dir(str(BASE_DIR / "temp/solutions"))

    # system initialization
    evolve = OpenEvolve(
        initial_program_path=str(BASE_DIR / "evolve/initial_program.txt"),
        evaluation_file=str(BASE_DIR / "evolve/evaluator.py"),
        config_path=str(BASE_DIR / "evolve/config.yaml"),
    )

    # latest checkpoint
    all_checkpoints = sorted((BASE_DIR / "evolve/openevolve_output/checkpoints").glob("checkpoint_*"))
    checkpoint_path = str(all_checkpoints[-1]) if all_checkpoints else None
    print(f"Using checkpoint: '{checkpoint_path}'.")

    # running evolution
    asyncio.run(main(evolve, checkpoint_path))