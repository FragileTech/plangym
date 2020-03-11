from setuptools import find_packages, setup


setup(
    name="plangym",
    description="OpenAI gym environments adapted for planning.",
    version="0.0.1",
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
        "atari-py<=0.2.0",
    ],
    packages=find_packages(),
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT license",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
    ],
)
