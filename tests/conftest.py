import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
STEM = "2018-03-13.17-05-00.17-10-00.school.G328"


@pytest.fixture
def act_path():
    return os.path.join(FIXTURES, f"{STEM}.activities.yml")


@pytest.fixture
def geom_path():
    return os.path.join(FIXTURES, f"{STEM}.geom.yml")
