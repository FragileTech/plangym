from setuptools import setup


setup(
    name="plangym",
    description="OpenAI gym environments adapted for planning.",
    version="0.0.1",
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="guillem.db@gmail.com",
    url="https://github.com/Guillemdb/plangym",
    download_url="https://github.com/Guillemdb/plangym",
    keywords=[
        "gym",
        "reinforcement learning",
        "artificial intelligence",
        "monte carlo",
        "planning",
    ],
    install_requires=["gym", "numpy", "Pillow-simd"],
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
