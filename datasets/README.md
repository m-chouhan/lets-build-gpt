# Datasets

Plain text files. Pick one with `--data_path`, e.g. `uv run python mini_gpt/train.py --data_path datasets/names.txt`.


| File              | What                                                | Source                                                                                                                                                                                                                            |
| ----------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shakespeare.txt` | Tiny Shakespeare, ~1.1M chars                       | [karpathy/char-rnn](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)                                                                                                                    |
| `names.txt`       | 32k first names, one per line, sorted by popularity | [karpathy/makemore](https://raw.githubusercontent.com/karpathy/makemore/master/names.txt)                                                                                                                                         |
| `python_code.txt` | 11 CPython 3.12.0 stdlib files joined, ~1.1M chars  | [python/cpython](https://github.com/python/cpython/tree/v3.12.0/Lib) `v3.12.0/Lib`: `_pydecimal.py typing.py argparse.py inspect.py dataclasses.py subprocess.py pathlib.py tarfile.py logging/__init__.py enum.py statistics.py` |


| `indian_names.txt` | 6.1k Indian first names (romanized, lowercase), one per line, sorted by frequency | [deedy's gist](https://gist.github.com/deedy/dfe64d21faa4180aff64495dc1c76f52), `Name` column of `indian_first_names.csv` |
| `hindi_names.txt` | `indian_names.txt` in Devanagari: 5.6k unique names (spelling variants like pooja/puja merge) | Transliterated with [AI4Bharat IndicXlit](https://github.com/AI4Bharat/IndicXlit) (`XlitEngine("hi", beam_width=4, rescore=False)`, top-1). Machine output, so a few names are off (e.g. priya → प्रिय) |
| `hindi_names_electoral.txt` | 34k first names in Devanagari, as written in Indian electoral rolls of Hindi-belt states (UP, Bihar, MP, Rajasthan, Haryana, Jharkhand, Uttarakhand, Himachal, Chandigarh). Names held by ≥100 voters, sorted by frequency | [Harvard Dataverse doi:10.7910/DVN/WZGJBM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WZGJBM), file `in_rolls_state_year_native_fn_naampy_x10.csv.gz` (CC0). Summed over states and birth years; kept only pure-Devanagari names, dropped abbreviations with digits (मो०, कु०) and stray Vedic stress marks |
