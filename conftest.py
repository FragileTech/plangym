import sys

collect_ignore = ["setup.py", "plangym/ray.py"]
if sys.version_info >= (3, 8):
    collect_ignore.append("plangym/retro.py")
    collect_ignore.append("tests/test_retro.py")
