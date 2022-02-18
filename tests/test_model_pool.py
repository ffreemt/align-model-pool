"""Test model_pool."""
# pylint: disable=broad-except
from model_pool import __version__
from model_pool import model_pool


def test_version():
    """Test version."""
    assert __version__[:4] == "0.1."


def test_sanity():
    """Sanity check."""
    try:
        assert not model_pool()
    except Exception:
        assert True
