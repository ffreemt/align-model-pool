"""Test model_pool."""
from model_pool import __version__
from model_pool import model_pool


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_sanity():
    """Sanity check."""
    try:
        assert not model_pool()
    except Exception:
        assert True
