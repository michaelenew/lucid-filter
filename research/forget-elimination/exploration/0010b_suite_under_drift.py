"""0010b -- the shipped test suite with `forget=None` made the default.

Wraps `LucidFilter.__init__` so that a construction which does not name `forget` gets the
class drift, then runs `lucid/tests/test_lucid.py`.  Tests that pass `forget` explicitly keep
their value.  Nothing on disk is changed.  The report is pytest's own (and its cache is the
checkpoint: `--lf` re-runs only what failed).

Run from the repository root:  python research/forget-elimination/exploration/0010b_suite_under_drift.py [pytest args]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import pytest
from lucid.filter import lucid as L
_init = L.LucidFilter.__init__
def _drift_default(self, *a, **kw):
    kw.setdefault("forget", None)
    return _init(self, *a, **kw)
L.LucidFilter.__init__ = _drift_default
sys.exit(pytest.main(["-q", "-rf", "--tb=line", "-p", "no:cacheprovider", "lucid/tests/test_lucid.py",
                      "--deselect", "lucid/tests/test_lucid.py::test_class_drift_is_opt_in"] + sys.argv[1:]))
