from unittest.mock import Mock

import pytest


@pytest.fixture()
def mock_generate_():
    return Mock(return_value=[1, 1, 1, 1])
