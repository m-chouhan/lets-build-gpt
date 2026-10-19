"""All the knobs in one place. Tune here, not inside the other files.

Every knob has a default here, and every knob can be overridden from the
command line with a flag of the same name: --seed 42 --eval_iters 100
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_args


@dataclass
class Config:
    # which model to run (see models/)
    model: str = "BigramCounts"

    # data
    data_path: Path = Path(__file__).parent.parent / "datasets" / "shakespeare.txt"  # any .txt works: --data_path my_text.txt
    train_frac: float = 0.9  # 90% train, 10% validation

    # batching (random chunks used for scoring)
    batch_size: int = 32  # B: chunks processed in parallel
    block_size: int = 8  # T: characters per chunk (max context)

    # evaluation
    eval_iters: int = 50  # batches averaged per loss estimate

    # generation
    gen_tokens: int = 500

    # randomness
    seed: int | None = 1337  # same random numbers every run; None (--seed false) = new seed each run

    @classmethod
    def from_cli(cls, **choices: list[str]) -> "Config":
        """Defaults from above, overridden by any --<knob> flags.

        choices limits a knob to known values, e.g. from_cli(model=["BigramCounts", ...])
        https://docs.python.org/3/library/argparse.html
        """
        parser = argparse.ArgumentParser()
        for f in fields(cls):
            parser.add_argument(f"--{f.name}", default=f.default, type=_parser_for(f), choices=choices.get(f.name))
        return cls(**vars(parser.parse_args()))


def _parser_for(f):
    """Turn the typed text into the knob's type: "0.01" -> 0.01, "false" -> None."""
    allows_none = type(None) in get_args(f.type)  # e.g. `int | None`
    to_type = type(f.default)  # int, float, str, Path

    def parse(text: str):
        if allows_none and text.lower() == "false":
            return None
        return to_type(text)

    parse.__name__ = to_type.__name__  # nicer error: "invalid int value: 'abc'"
    return parse
