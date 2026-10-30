import pytest

from sfs.data.sampling import n_samples, sample_frames


@pytest.mark.parametrize("frames,expected", [
    (8, 1),      # 0.27 s -> 1
    (32, 3),     # p25 instance at 30 fps -> ceil(2.13) = 3
    (67, 5),     # median -> ceil(4.47) = 5
    (144, 10),   # p75 -> ceil(9.6) = 10
    (9009, 16),  # 5 min -> capped
])
def test_n_samples_matches_spec_table(frames, expected):
    assert n_samples(0, frames - 1, fps=30) == expected


def test_single_sample_is_midpoint():
    assert sample_frames(100, 106, fps=30) == [103]


def test_endpoints_included_sorted_unique_in_range():
    f = sample_frames(1000, 1143, fps=30)
    assert f[0] == 1000 and f[-1] == 1143
    assert f == sorted(set(f))
    assert all(1000 <= x <= 1143 for x in f)


def test_cap_respected_for_long_instances():
    assert len(sample_frames(0, 100_000, fps=30)) == 16


def test_rate_and_cap_are_configurable():
    assert n_samples(0, 299, fps=30, rate=1.0) == 10
    assert n_samples(0, 299, fps=30, rate=4.0, n_max=8) == 8


def test_invalid_inputs():
    with pytest.raises(ValueError):
        n_samples(10, 5, fps=30)
    with pytest.raises(ValueError):
        n_samples(0, 5, fps=0)
