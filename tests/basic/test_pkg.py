from importlib import import_module
from pathlib import Path

import pytest

from PyCXpress import get_include


@pytest.mark.parametrize(
    ("include_suffix",),
    [
        ("PyCXpress/include",),
    ],
)
def test_get_include(include_suffix):
    assert get_include().endswith(include_suffix)
