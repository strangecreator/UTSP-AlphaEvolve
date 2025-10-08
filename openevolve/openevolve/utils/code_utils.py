"""
Utilities for code parsing, diffing, and manipulation
"""

import re
from typing import Dict, List, Optional, Tuple, Union


def _normalize_newlines(text: str) -> str:
    """
    Convert CRLF/CR to LF. Do not strip any other whitespace.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    code = _normalize_newlines(code)
    lines = code.split("\n")
    blocks: List[Tuple[int, int, str]] = []

    in_block = False
    start_line = -1
    block_content: List[str] = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text, tolerant to CRLF and extra spaces.

    Format:
        <<<<<<< SEARCH
        ...search...
        =======
        ...replace...
        >>>>>>> REPLACE
    """
    diff_text = _normalize_newlines(diff_text)

    # Allow optional spaces after markers and tolerate CRLF already normalized.
    # The replace section may or may not end with a newline before the >>>>>>> line.
    pattern = re.compile(
        r"""
        <<<<<<<\s*SEARCH[ \t]*\n     # start marker
        (.*?)                        # search block (non-greedy)
        \n=======[ \t]*\n            # separator
        (.*?)                        # replace block (non-greedy)
        \n>>>>>>>[ \t]*REPLACE       # end marker
        """,
        re.DOTALL | re.VERBOSE,
    )

    blocks = pattern.findall(diff_text)

    # Do not strip inner whitespace; only strip a single leading/trailing newline
    # caused by the regex boundaries so substring matches remain faithful.
    res: List[Tuple[str, str]] = []
    for search, replace in blocks:
        # Normalize only the outermost newlines introduced by the pattern
        if search.startswith("\n"):
            search = search[1:]
        if replace.startswith("\n"):
            replace = replace[1:]
        res.append((search, replace))
    return res


def _lines_equal_strict(a: List[str], b: List[str]) -> bool:
    return a == b


def _lines_equal_rstrip(a: List[str], b: List[str]) -> bool:
    return [x.rstrip() for x in a] == [x.rstrip() for x in b]


def _norm_ws(s: str) -> str:
    # Collapse runs of spaces/tabs, preserve other chars
    return re.sub(r"[ \t]+", " ", s.rstrip())


def _lines_equal_ws_norm(a: List[str], b: List[str]) -> bool:
    return [_norm_ws(x) for x in a] == [_norm_ws(y) for y in b]


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply SEARCH/REPLACE diff blocks to the original code.
    Robust to CRLF vs LF and minor whitespace differences.

    Strategy per block:
      1) strict match
      2) trailing-space-insensitive match
      3) whitespace-normalized match (space/tab runs collapsed)
    """
    original_code = _normalize_newlines(original_code)
    result_lines = original_code.split("\n")

    diff_blocks = extract_diffs(diff_text)
    if not diff_blocks:
        raise ValueError(
            "No diff blocks found. Ensure your diff is formatted as:\n"
            "<<<<<<< SEARCH\\n...\\n=======\\n...\\n>>>>>>> REPLACE"
        )

    for idx, (search_text, replace_text) in enumerate(diff_blocks, 1):
        search_text = _normalize_newlines(search_text)
        replace_text = _normalize_newlines(replace_text)

        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        m = len(search_lines)
        if m == 0:
            raise ValueError(f"Empty SEARCH block in diff #{idx}.")

        found_at: Optional[int] = None

        # Pass 1: strict equality
        for i in range(len(result_lines) - m + 1):
            if _lines_equal_strict(result_lines[i : i + m], search_lines):
                found_at = i
                break

        # Pass 2: trailing-space-insensitive
        if found_at is None:
            for i in range(len(result_lines) - m + 1):
                if _lines_equal_rstrip(result_lines[i : i + m], search_lines):
                    found_at = i
                    break

        # Pass 3: whitespace-normalized (collapse runs of spaces/tabs)
        if found_at is None:
            for i in range(len(result_lines) - m + 1):
                if _lines_equal_ws_norm(result_lines[i : i + m], search_lines):
                    found_at = i
                    break

        if found_at is None:
            preview = search_lines[0] if search_lines else "<empty>"
            raise ValueError(
                f"SEARCH block #{idx} not found in target source.\n"
                f"First SEARCH line: {preview!r}\n"
                "Hint: check for small whitespace differences or CRLF/LF issues."
            )

        # Replace
        result_lines[found_at : found_at + m] = replace_lines

    return "\n".join(result_lines)


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    llm_response = _normalize_newlines(llm_response)

    # Try exact language first
    code_block_pattern = r"```" + re.escape(language) + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any fenced block
    code_block_pattern = r"```(?:[\w.+-]+)?\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary: List[str] = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    code1 = _normalize_newlines(code1)
    code2 = _normalize_newlines(code2)

    if code1 == code2:
        return 0

    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost,   # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    code = _normalize_newlines(code)

    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"