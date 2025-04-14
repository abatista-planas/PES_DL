from unittest.mock import Mock

import pytest

from pes_1D import collect




def test_collect_samples():
    want = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1],
    ]
    got = collect.collect_samples(4, 1)
    assert got == want



def test_generate_sample():
    print("MY test_collect.py")
    print(collect.__dict__)
    want = [3, 3, 3, 3, 3]
    got = collect.generate_sample(2, 1)
    assert got == want


# REFACTOR 1
def test_collect_samples_1(monkeypatch):
    def mock_generate(sample_number, base):
        return [1, 1, 1, 1]

    monkeypatch.setattr("pes_1D.collect.generate_sample", mock_generate)

    want = [[1, 1, 1, 1] for _ in range(4)]
    got = collect.collect_samples(4, 1)
    assert got == want


# REFACTOR 2
def test_collect_samples_2(monkeypatch):
    monkeypatch.setattr(
        "pes_1D.collect.generate_sample", Mock(return_value=[1, 1, 1, 1])
    )  # noqa
    want = [[1, 1, 1, 1] for _ in range(4)]
    got = collect.collect_samples(4, 1)
    assert got == want


# REFACTOR 3
@pytest.fixture()
def mock_generate():
    return Mock(return_value=[1, 1, 1, 1])


def test_collect_samples_3(monkeypatch, mock_generate_):
    monkeypatch.setattr("pes_1D.collect.generate_sample", mock_generate_)
    want = [[1, 1, 1, 1] for _ in range(4)]
    got = collect.collect_samples(4, 1)
    assert got == want
    assert mock_generate_.called