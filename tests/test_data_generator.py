import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pes_1D import DataGenerator

# def test_lennard_jones_pes_empty():
#     want = [
#         [1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [1, 1, 1, 1, 1],
#     ]
#     got = collect.collect_samples(4, 1)
#     assert got == want


def test_exception_message_lennard_jones_pes():
    params = [
        [1.0, 0.5, -1, 1, 10],
        [1.0, 0.5, 1, -1, 10],
        [1.0, 0.5, 1, 1, -10],
        [1.0, 0.5, 2, 1, 10],
    ]
    for sigma, epsilon, R_min, R_max, size in params:
        with pytest.raises(Exception) as e_info:
            DataGenerator.lennard_jones_pes(sigma, epsilon, R_min, R_max, size)
        assert str(e_info.value) == "Size and range must be positive"


def test_lennard_jones_pes():
    df_want = pd.DataFrame(
        {
            "R": [
                1.0,
                1.1111111111111112,
                1.2222222222222223,
                1.3333333333333333,
                1.4444444444444444,
                1.5555555555555556,
                1.6666666666666665,
                1.7777777777777777,
                1.8888888888888888,
                2.0,
            ],
            "Energy": [
                0.0,
                -0.4980229270379999,
                -0.4199876714144228,
                -0.29260432720184326,
                -0.1959589943368627,
                -0.1311983755954567,
                -0.08895843532800009,
                -0.06134592149305007,
                -0.04306483161404184,
                -0.03076171875,
            ],
        }
    )

    df_get = DataGenerator.lennard_jones_pes(1.0, 0.5, 1.0, 2.0, 10)

    assert_frame_equal(df_get, df_want, atol=1e-16)
