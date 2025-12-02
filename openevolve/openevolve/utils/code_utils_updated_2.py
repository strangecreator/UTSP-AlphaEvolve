"""
Utilities for code parsing, diffing, and manipulation.

This version adds a 4th pass in apply_diff:
    1) strict match
    2) trailing-space-insensitive match
    3) whitespace-normalized match
    4) FUZZY line-window match (SequenceMatcher / rapidfuzz if installed)

The fuzzy step is designed specifically to mitigate small LLM inconsistencies:
- extra/missing blank lines,
- slightly renamed variables in context lines,
- small formatting shifts.

IMPORTANT: fuzzy step is last and guarded by a threshold to avoid wild replaces.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union, Callable

# optional acceleration
try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except Exception:  # noqa: BLE001
    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_newlines(text: str) -> str:
    """
    Convert CRLF/CR to LF. Do not strip any other whitespace.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _norm_ws(s: str) -> str:
    # Collapse runs of spaces/tabs, preserve other chars
    return re.sub(r"[ \t]+", " ", s.rstrip())


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code:

        # EVOLVE-BLOCK-START
        ...
        # EVOLVE-BLOCK-END

    Returns: list of (start_line_idx, end_line_idx, content_without_markers)
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

    res: List[Tuple[str, str]] = []
    for search, replace in blocks:
        if search.startswith("\n"):
            search = search[1:]
        if replace.startswith("\n"):
            replace = replace[1:]
        res.append((search, replace))
    return res


# ---------------------------------------------------------------------------
# Line comparison strategies
# ---------------------------------------------------------------------------

def _lines_equal_strict(a: List[str], b: List[str]) -> bool:
    return a == b


def _lines_equal_rstrip(a: List[str], b: List[str]) -> bool:
    return [x.rstrip() for x in a] == [x.rstrip() for x in b]


def _lines_equal_ws_norm(a: List[str], b: List[str]) -> bool:
    return [_norm_ws(x) for x in a] == [_norm_ws(y) for y in b]


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    """
    Return similarity in [0, 1].
    Uses rapidfuzz if available, else difflib.
    """
    if _HAS_RAPIDFUZZ:
        # rapidfuzz returns 0..100
        return _rf_fuzz.ratio(a, b) / 100.0
    else:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


def _find_best_fuzzy_window(
    haystack_lines: List[str],
    needle_lines: List[str],
    *,
    max_pad: int = 2,
) -> Tuple[Optional[int], float, Tuple[int, int]]:
    """
    Try to locate `needle_lines` inside `haystack_lines` approximately.

    We do a sliding-window search:
      - base window size = len(needle_lines)
      - we also try windows up to +max_pad lines bigger (to absorb extra blank lines)
    We return the best window start index and its similarity.

    Returns:
        (best_start_idx | None, best_ratio, (best_len, window_len))
    """
    n = len(haystack_lines)
    m = len(needle_lines)
    if m == 0:
        return None, 0.0, (0, 0)

    needle_text = "\n".join(needle_lines)

    best_idx: Optional[int] = None
    best_ratio = 0.0
    best_sizes = (m, m)

    # We allow slightly longer windows to absorb minor LLM noise
    for extra in range(0, max_pad + 1):
        win_len = m + extra
        if win_len > n:
            break

        for i in range(0, n - win_len + 1):
            window_text = "\n".join(haystack_lines[i : i + win_len])
            ratio = _similarity(window_text, needle_text)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
                best_sizes = (m, win_len)

    return best_idx, best_ratio, best_sizes


# ---------------------------------------------------------------------------
# Main diff applier
# ---------------------------------------------------------------------------

def apply_diff(
    original_code: str,
    diff_text: str,
    *,
    enable_fuzzy: bool = True,
    fuzzy_threshold: float = 0.84,
    fuzzy_max_pad: int = 2,
) -> str:
    """
    Apply SEARCH/REPLACE diff blocks to the original code.

    Strategy per block:
      1) strict match
      2) trailing-space-insensitive match
      3) whitespace-normalized match
      4) FUZZY line-window match (if enable_fuzzy=True)

    Fuzzy step:
      - we look for the most similar window of lines
      - if similarity >= fuzzy_threshold, we replace that window
      - else we fail (raise ValueError)

    Params:
        original_code: current file content
        diff_text: text containing <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
        enable_fuzzy: turn on/off the 4th pass
        fuzzy_threshold: similarity in [0,1]
        fuzzy_max_pad: allow window to be up to N lines longer than search

    Raises:
        ValueError if some SEARCH block cannot be placed.
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

        # Pass 3: whitespace-normalized
        if found_at is None:
            for i in range(len(result_lines) - m + 1):
                if _lines_equal_ws_norm(result_lines[i : i + m], search_lines):
                    found_at = i
                    break

        # Pass 4: fuzzy window (line-based)
        if found_at is None and enable_fuzzy:
            best_idx, best_ratio, (search_len, window_len) = _find_best_fuzzy_window(
                result_lines,
                search_lines,
                max_pad=fuzzy_max_pad,
            )
            if best_idx is not None and best_ratio >= fuzzy_threshold:
                # We replace the actual window_len lines, not just search_len
                # because the window may include extra blank lines or minor noise.
                # This is the whole point of fuzzy mitigation.
                found_at = best_idx
                m = window_len
            else:
                preview = search_lines[0] if search_lines else "<empty>"
                raise ValueError(
                    f"SEARCH block #{idx} not found in target source, even with fuzzy match.\n"
                    f"Best fuzzy ratio: {best_ratio:.3f} (threshold {fuzzy_threshold})\n"
                    f"First SEARCH line: {preview!r}\n"
                    "Hint: lower fuzzy_threshold or increase fuzzy_max_pad."
                )

        if found_at is None:
            # Should not reach here, but keep for safety
            preview = search_lines[0] if search_lines else "<empty>"
            raise ValueError(
                f"SEARCH block #{idx} not found in target source.\n"
                f"First SEARCH line: {preview!r}"
            )

        # Finally, replace
        result_lines[found_at : found_at + m] = replace_lines

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Other helpers (kept for completeness)
# ---------------------------------------------------------------------------

def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response.
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
    (simple DP, fine for small snippets).
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
    Primitive language detection
    """
    code = _normalize_newlines(code)

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