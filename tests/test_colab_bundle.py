"""The generated Colab bundle must reconstruct the package and import from its own copy."""
import importlib
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
import build_colab_bundle  # noqa: E402


def test_bundle_reconstructs_and_imports(tmp_path, monkeypatch):
    out = build_colab_bundle.build(str(tmp_path / "bundle.py"))
    root = tmp_path / "unpacked"
    monkeypatch.setenv("SFS_BUNDLE_ROOT", str(root))
    saved = {k: v for k, v in sys.modules.items() if k == "sfs" or k.startswith("sfs.")}
    try:
        runpy.run_path(out)
        mod = importlib.import_module("sfs.data.sampling")
        assert os.path.abspath(mod.__file__).startswith(str(root))
        assert mod.n_samples(0, 66, 30) == 5
        assert set(build_colab_bundle.collect()) <= {
            os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
            for d, _, fs in os.walk(root) for f in fs}
    finally:
        sys.path[:] = [p for p in sys.path if p != str(root)]
        for k in [k for k in sys.modules if k == "sfs" or k.startswith("sfs.")]:
            del sys.modules[k]
        sys.modules.update(saved)
