"""Test load_model model-l."""
from model_pool.load_model import load_model
from logzero import logger


def test_model_l1():
    """Test model-l 1."""
    try:
        clas = load_model("model-l", alive_bar_on=False)
    except Exception as exc:
        logger.error("load_model('model-l'): %s", exc)
        # raise SystemExit(1) from exc
        assert False, str(exc)

    res = clas("test", ["test", "测试"], multi_lable=True)
    assert res.keys() == ["labels", "scores", "sequence"]
