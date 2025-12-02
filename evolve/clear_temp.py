import sys
import shutil
import argparse
import datetime as dt
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete subdirectories whose names contain a timestamp "
            'in format "YYYY_MM_DD-HH_MM_SS-<uuid>" older than N minutes.'
        )
    )
    parser.add_argument(
        "root_dir",
        help="Absolute path to the directory that contains timestamped subdirectories.",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=30,
        help="Age threshold in minutes (default: 30).",
    )
    return parser.parse_args()


def parse_timestamp_from_name(name: str) -> dt.datetime | None:
    """
    Expected name format:
        'YYYY_MM_DD-HH_MM_SS-...'
    Example:
        '2025_11_15-11_05_22-de04f3bc-74de-4167-b7c6-71a6e159c2b3'
    We take the first 19 characters: 'YYYY_MM_DD-HH_MM_SS'
    and parse with strptime.
    """
    # Minimum length must cover 'YYYY_MM_DD-HH_MM_SS'
    if len(name) < 19:
        return None

    time_str = name[:19]  # '2025_11_15-11_05_22'
    try:
        return dt.datetime.strptime(time_str, "%Y_%m_%d-%H_%M_%S")
    except ValueError:
        return None


def should_delete(ts: dt.datetime, now: dt.datetime, minutes: int) -> bool:
    """
    Return True if directory timestamp 'ts' is at least 'minutes' older than 'now'.

    Condition:
        (now - ts) ≥ minutes
    In code:
        (now - ts).total_seconds() >= minutes * 60
    """
    delta = now - ts
    return delta.total_seconds() >= minutes * 60


def main() -> int:
    args = parse_args()

    root = Path(args.root_dir).expanduser()
    if not root.exists():
        print(f"ERROR: Root directory does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"ERROR: Not a directory: {root}", file=sys.stderr)
        return 1

    now = dt.datetime.now()
    minutes = args.minutes

    print(f"Root directory: {root}")
    print(f"Threshold: {minutes} minutes")
    print(f"Current time: {now}")
    print("-" * 60)

    deleted = 0
    skipped = 0
    failed = 0

    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        ts = parse_timestamp_from_name(entry.name)
        if ts is None:
            print(f"SKIP (no valid timestamp): {entry.name}")
            skipped += 1
            continue

        if should_delete(ts, now, minutes):
            try:
                shutil.rmtree(entry)
                print(f"DELETE: {entry.name} (timestamp={ts})")
                deleted += 1
            except Exception as e:
                print(f"ERROR deleting {entry.name}: {e}", file=sys.stderr)
                failed += 1
        else:
            print(f"KEEP:   {entry.name} (timestamp={ts})")
            skipped += 1

    print("-" * 60)
    print(f"Deleted: {deleted}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())