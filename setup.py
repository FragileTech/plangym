from setuptools import find_packages, setup
from plangym.version import __version__

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
        "Pillow>=7.0.0",
        "opencv-python>=4.2.0.32",
        "pytest>=4.0.1",
        "atari-py==0.1.1",
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
