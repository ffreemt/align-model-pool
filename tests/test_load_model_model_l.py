"""Test load_model model-l."""
# pylint: disable=broad-except
from logzero import logger
from model_pool.load_model import load_model


def test_model_l1():
    """Test model-l 1."""
    try:
        clas = load_model("model-l", alive_bar_on=False)
    except Exception as exc:
        logger.error("load_model('model-l'): %s", exc)
        # raise SystemExit(1) from exc
        assert False, str(exc)

    # res = clas("test", ["tests", "love"], multi_lable=True)
    res = clas("Liebe", ["test", "machen", "love"], multi_lable=False)

    # need to take order into account
    # assert res.keys() == ["labels", "scores", "sequence"]

    assert res.get('scores')[0] > 0.8  # 0.87
