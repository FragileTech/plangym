from setuptools import find_packages, setup
from plangym.version import __version__

extras = {
    "atari": ["atari-py==0.1.1"],
    "retro": ["gym-retro>=0.7.0"],
    "test": ["pytest>=5.3.5"],
    "ray": ["ray", "setproctitle"],
}

extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="plangym",
    description="OpenAI gym environments adapted for planning.",
    version=__version__,
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="guillem.db@fragile.tech",
    url="https://github.com/FragileTech/plangym",
    download_url="https://github.com/FragileTech/plangym",
    keywords=[
        "gym",
        "reinforcement learning",
        "artificial intelligence",
        "monte carlo",
        "planning",
        "atari games",
    ],
    install_requires=[
        "numpy>=1.16.2",
        "gym>=0.10.9",
        "pillow-simd>=7.0.0.post3",
        "opencv-python>=4.2.0.32",
    ],
    packages=find_packages(),
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
    ],
)
