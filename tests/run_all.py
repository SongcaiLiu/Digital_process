"""Run every project regression test without requiring pytest."""
import importlib
import pkgutil
import tests

modules=sorted(
    item.name for item in pkgutil.iter_modules(tests.__path__)
    if item.name.startswith("test_")
)
for name in modules:
    importlib.import_module(f"tests.{name}")
print(f"ALL PASS: {len(modules)} test modules")
