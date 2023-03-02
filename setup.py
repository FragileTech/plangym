"""plangym package installation metadata."""
from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "plangym.version",
    str(Path(__file__).parent / "plangym" / "version.py"),
).load_module()

with open(Path(__file__).with_name("README.md"), encoding="utf-8") as f:
    long_description = f.read()

extras = {
    "atari": ["ale-py>=0.7.0", "autorom[accept-rom-license]==0.4.2"],
    "retro": ["gym-retro>=0.8.0"],
    "test": ["pytest>=5.3.5", "pyvirtualdisplay>=1.3.2"],
    "ray": ["ray", "setproctitle"],
    "box2d": ["box2d-py==2.3.5", "pyglet>=1.4.0"],
}

extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="plangym",
    description="Plangym is an interface to use OpenAI gym for planning problems. It extends the standard interface to allow setting and recovering the environment states.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version=version.__version__,
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="info@fragile.tech",
    url="https://github.com/FragileTech/plangym",
    keywords=["Machine learning", "artificial intelligence"],
    test_suite="tests",
    tests_require=["pytest>=5.3.5", "hypothesis>=5.6.0"],
    install_requires=[
        "numpy",
        "pillow==8.4.0",
        "opencv-python>=4.2.0.32",
    ],
    extras_require=extras,
    package_data={"": ["README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
    ],
)
