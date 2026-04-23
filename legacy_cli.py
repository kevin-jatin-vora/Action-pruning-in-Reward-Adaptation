from __future__ import annotations

import argparse


def parse_seed_range(
    start_prompt: str = "start: ",
    end_prompt: str = "end: ",
    default_start: int | None = None,
    default_end: int | None = None,
) -> tuple[int, int]:
    """Parse a seed range from CLI flags with an interactive fallback."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start", type=int, default=default_start)
    parser.add_argument("--end", type=int, default=default_end)
    args, _ = parser.parse_known_args()

    start_index = args.start if args.start is not None else int(input(start_prompt))
    end_index = args.end if args.end is not None else int(input(end_prompt))

    if end_index <= start_index:
        raise ValueError(
            f"Expected end index greater than start index, got start={start_index}, end={end_index}."
        )

    return start_index, end_index
