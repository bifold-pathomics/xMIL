# Tests

Automated software tests should be added here. The root-level `test.py` file is a
model evaluation command rather than a software test suite.

## Survival checkpoint windows

Run `PYTHONPATH=src pytest tests/test_survival_windows.py` to check full-window
statistics, selected versus raw metrics, checkpoint persistence, legacy criteria,
and validation intervals on CPU.
