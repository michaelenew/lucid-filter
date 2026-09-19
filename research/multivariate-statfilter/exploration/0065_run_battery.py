"""Run a research rig's main() with the shipped filter swapped for a given source (e.g. `git show <rev>:lucid/filter/lucid.py`).
    python 0065_run_battery.py <lucid.py source|tree> <rig.py>"""
import sys, types, importlib.util, os
sys.path.insert(0, ".")
src, rig = sys.argv[1], sys.argv[2]
if src != "tree":
    import lucid.filter
    mod = types.ModuleType("lucid.filter.lucid"); mod.__file__ = "lucid/filter/lucid.py"; mod.__package__ = "lucid.filter"
    sys.modules["lucid.filter.lucid"] = mod
    exec(compile(open(src).read(), "lucid/filter/lucid.py", "exec"), mod.__dict__)
    lucid.filter.lucid = mod
    import lucid as _top
    for name in ("LucidFilter", "LucidResult", "LucidStep"):
        if hasattr(mod, name):
            setattr(lucid.filter, name, getattr(mod, name))
            if hasattr(_top, name): setattr(_top, name, getattr(mod, name))
    from lucid import LucidFilter as _chk
    assert not hasattr(_chk(), "memories"), "swap did not reach `lucid.LucidFilter`"
    print("filter source:", src, "| memory ladder:", hasattr(mod, "_memory_rungs"), flush=True)
else:
    import lucid.filter.lucid as L; print("filter source: tree | memory ladder:", hasattr(L, "_memory_rungs"), flush=True)
sys.path.insert(0, os.path.dirname(rig)); sys.path.insert(0, os.path.join(os.path.dirname(rig), "..", "scripts"))
spec = importlib.util.spec_from_file_location("rigmain", rig); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.main()
