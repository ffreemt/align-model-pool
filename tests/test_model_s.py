"""Test fetch_check_aux."""
from model_pool.model_s import load_model_s

model_s = load_model_s(alive_bar_on=False)


def test_model_s1():
    """Test model_s 1"""
    assert model_s.encode("a").shape[0] == 512
    assert model_s.encode(["a", "b"]).shape == (2, 512)
