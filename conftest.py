import numpy
import pytest

import plangym


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["plangym"] = plangym
