# UMINT cv09

Python rewrite of the provided MATLAB crossroad-control assignment.

## Main entry point

```bash
python3 /Users/gnomik7/PycharmProjects/UMINT_cv09/main.py
```

This default run compares the fixed controller with the fuzzy controller in modes `2..6` and writes CSV/SVG artifacts to:

```text
/Users/gnomik7/PycharmProjects/UMINT_cv09/python_outputs
```

## Useful commands

Run only the fuzzy controller in mode 6:

```bash
python3 /Users/gnomik7/PycharmProjects/UMINT_cv09/main.py --controller fuzzy --mode 6
```

Run only the fixed controller in mode 6:

```bash
python3 /Users/gnomik7/PycharmProjects/UMINT_cv09/main.py --controller fixed --mode 6
```

Print the phase-by-phase trace for fuzzy mode 6:

```bash
python3 /Users/gnomik7/PycharmProjects/UMINT_cv09/main.py --controller fuzzy --mode 6 --trace-phases
```

## Project layout

- `main.py` — Python CLI entry point
- `python_crossroad/` — rewritten simulator, fuzzy controller, reporting, scenario data
- `ulohy/` — original assignment materials, MATLAB sources, and the written report
