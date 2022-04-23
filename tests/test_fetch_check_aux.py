"""Test fetch_check_aux."""
from model_pool import fetch_check_aux
from logzero import logger


def test_fetch_check_aux1():
    """Test fetch_check_aux 1"""
    try:
        res = fetch_check_aux("/root")
    except PermissionError:
        # may not be able to write to /root
        # write to ~
        res = fetch_check_aux()
    except Exception:
        logger.exception("")
        raise

    assert res
