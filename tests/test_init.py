
import importlib

import compitum as inner_compitum


def test_init_all() -> None:
    """Tests that all modules in compitum.__all__ are importable."""
    assert hasattr(inner_compitum, "__all__")
    for module_name in inner_compitum.__all__:
        # These modules are relative to the 'compitum' package
        importlib.import_module(f"compitum.{module_name}")
