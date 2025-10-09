import sys
import shutil
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "evolve"))

# other imports
from utils import *


JSON_HPP_PATH = str(BASE_DIR / "UTSP/include/json.hpp")


TEMPLATE = """
* heat_map_train.py *:
@@@
{heat_map_train_py_code}
@@@

* heat_map_inference.py *:
@@@
{heat_map_inference_py_code}
@@@

* include/utils.hpp *:
@@@
{utils_hpp_code}
@@@

* include/context.hpp *:
@@@
{context_hpp_code}
@@@

* include/random_solution.hpp *:
@@@
{random_solution_hpp_code}
@@@

* include/local_2_opt_search.hpp *:
@@@
{local_2_opt_search_hpp_code}
@@@

* include/local_k_opt_search.hpp *:
@@@
{local_k_opt_search_hpp_code}
@@@

* include/additional.hpp *:
@@@
{additional_hpp_code}
@@@

* TSP.cpp *:
@@@
{TSP_cpp_code}
@@@

* config.json *:
@@@
{config_json_code}
@@@

* changes_description.txt *:
@@@
{changes_description_txt_code}
@@@
""".strip()

DEFAULT_PARSE_ERROR = "Could not parse code (maybe not present). If you have provided a new code for this file then, check on correct formatting and filenames! You can write only in those 10 files. (\"heat_map_train.py\", \"heat_map_inference.py\", \"config.json\", \"TSP.cpp\", and 6 .hpp's in the include/ folder)"

NAMES_TO_FILENAMES = {
    "heat_map_train_py_code": "heat_map_train.py",
    "heat_map_inference_py_code": "heat_map_inference.py",
    "utils_hpp_code": "include/utils.hpp",
    "context_hpp_code": "include/context.hpp",
    "random_solution_hpp_code": "include/random_solution.hpp",
    "local_2_opt_search_hpp_code": "include/local_2_opt_search.hpp",
    "local_k_opt_search_hpp_code": "include/local_k_opt_search.hpp",
    "additional_hpp_code": "include/additional.hpp",
    "TSP_cpp_code": "TSP.cpp",
    "config_json_code": "config.json",
    "changes_description_txt_code": "changes_description.txt",
}


def format_query_code(dir_path: str) -> str:
    return TEMPLATE.format(
        heat_map_train_py_code=read_file(f"{dir_path}/heat_map_train.py"),
        heat_map_inference_py_code=read_file(f"{dir_path}/heat_map_inference.py"),
        utils_hpp_code=read_file(f"{dir_path}/include/utils.hpp"),
        context_hpp_code=read_file(f"{dir_path}/include/context.hpp"),
        random_solution_hpp_code=read_file(f"{dir_path}/include/random_solution.hpp"),
        local_2_opt_search_hpp_code=read_file(f"{dir_path}/include/local_2_opt_search.hpp"),
        local_k_opt_search_hpp_code=read_file(f"{dir_path}/include/local_k_opt_search.hpp"),
        additional_hpp_code=read_file(f"{dir_path}/include/additional.hpp"),
        TSP_cpp_code=read_file(f"{dir_path}/TSP.cpp"),
        config_json_code=read_file(f"{dir_path}/config.json"),
        changes_description_txt_code=read_file(f"{dir_path}/changes_description.txt"),
    )


def parse_output_code(text: str) -> None:
    def parse_code_block(text: str, block_header: str) -> str | None:
        text_splitted = text.split(block_header, maxsplit=1)

        if len(text_splitted) < 2: return None

        block_code = text_splitted[1].strip()

        if not block_code.startswith("@@@\n"): return None
        
        block_code = block_code.removeprefix("@@@\n").strip()

        if "\n@@@" not in block_code: return None

        return block_code.split("\n@@@", maxsplit=1)[0].strip()


    return {
        "heat_map_train_py_code": default_for_none(parse_code_block(text, "* heat_map_train.py *:"), DEFAULT_PARSE_ERROR),
        "heat_map_inference_py_code": default_for_none(parse_code_block(text, "* heat_map_inference.py *:"), DEFAULT_PARSE_ERROR),
        "utils_hpp_code": default_for_none(parse_code_block(text, "* include/utils.hpp *:"), DEFAULT_PARSE_ERROR),
        "context_hpp_code": default_for_none(parse_code_block(text, "* include/context.hpp *:"), DEFAULT_PARSE_ERROR),
        "random_solution_hpp_code": default_for_none(parse_code_block(text, "* include/random_solution.hpp *:"), DEFAULT_PARSE_ERROR),
        "local_2_opt_search_hpp_code": default_for_none(parse_code_block(text, "* include/local_2_opt_search.hpp *:"), DEFAULT_PARSE_ERROR),
        "local_k_opt_search_hpp_code": default_for_none(parse_code_block(text, "* include/local_k_opt_search.hpp *:"), DEFAULT_PARSE_ERROR),
        "additional_hpp_code": default_for_none(parse_code_block(text, "* include/additional.hpp *:"), DEFAULT_PARSE_ERROR),
        "TSP_cpp_code": default_for_none(parse_code_block(text, "* TSP.cpp *:"), DEFAULT_PARSE_ERROR),
        "config_json_code": default_for_none(parse_code_block(text, "* config.json *:"), DEFAULT_PARSE_ERROR),
        "changes_description_txt_code": default_for_none(parse_code_block(text, "* changes_description.txt *:"), DEFAULT_PARSE_ERROR),
    }


def save_parsed_output_code(parsed_code: dict, dir_path: str, include_json_hpp: bool = True) -> None:
    for name, block_code in parsed_code.items():
        filename = NAMES_TO_FILENAMES[name]

        file_path = f"{dir_path}/{filename}"
        touch_file(file_path)

        with open(file_path, 'w') as file:
            file.write(block_code)
    
    if include_json_hpp:
        shutil.copy2(JSON_HPP_PATH, f"{dir_path}/include")


if __name__ == "__main__":
    DIR_PATH = str(BASE_DIR / "UTSP")

    query_code = format_query_code(DIR_PATH)
    
    with open(str(BASE_DIR / "temp/query_code_sample.txt"), 'w') as file:
        file.write(query_code)
    
    parsed_output_code_dir = str(BASE_DIR / "temp/output_code_sample")
    create_dir(parsed_output_code_dir)

    parsed = parse_output_code(query_code)
    save_parsed_output_code(parsed, parsed_output_code_dir)