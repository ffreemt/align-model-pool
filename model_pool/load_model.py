"""Load model_l, based on model_s in radiobee-aliger.

return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)
"/root/.cache/huggingface/transformers/d23ec1d4898d7173d13994fbd662fa3243bf8a23744748d21115317316bd5e1b.a89eb3c0add0e1b04b46be11a1bc1a65b92fdae1bbb04124701ff2e6acfccc75"

do as admin in Windows or root in Liux
fetch_check_aux(default="/root")

"""
# pylint: disable=invalid-name, wrong-import-position, wrong-import-order, duplicate-code

_ = """
from install import install

_ = [
    *map(
        install,
        [
            "torch",
            "transformers",
            "sentencepiece",
            "huggingface-hub",
            "alive-progress",
            "logzero",
        ],
    )
]
# """

from pathlib import Path

import joblib
from huggingface_hub import hf_hub_url, cached_download  # hf_hub_download
from alive_progress import alive_bar
from logzero import logger

from model_pool.fetch_check_aux import fetch_check_aux

# prepare aux file for "model-l", this does not seem to work on gh workflow
try:
    # fetch_check_aux("/home/user")  # for hf spaces?
    fetch_check_aux("/root")
except PermissionError:  # for gh workflow
    logger.warning("You ll need to run as root or admin or give write permission to /root/.cache/huggingface or C:\root\.cache\huggingface to this user. Trying fetch_check_aux('~')")
    try:
        fetch_check_aux("~")
    except Exception as _:
        logger.error("fetch_check_aux('~') exc: %s", _)
        raise 
except Exception as e:
    logger.error(" fetch_check_aux() exc: %s", e)
    raise


def load_model(model_name, dir_loc=None, alive_bar_on=True):
    """Load local model_name=model_s if present, else fetch from hf.co."""
    if dir_loc is None:
        dir_loc = ""
    dir_loc = Path(dir_loc).absolute().as_posix()
    file_loc = f"{dir_loc}/{model_name}"

    if Path(file_loc).exists():
        if alive_bar_on:
            with alive_bar(
                1,
                title=f" Loading {dir_loc}/{model_name}, takes ~30 secs ...",
                length=3,
            ) as progress_bar:
                model = joblib.load(file_loc)

                # model_s = pickle.load(open(file_loc, "rb"))
                progress_bar()  # pylint: disable=not-callable
        else:
            logger.info("Loading %s/%s, takes ~30 secs ...", dir_loc, model_name)
            model = joblib.load(file_loc)
    else:
        logger.info(
            "Fetching and caching %s from huggingface.co... "
            "The first time may take a while depending on your net.",
            model_name,
        )
        if alive_bar_on:
            with alive_bar(
                1, title=" Subsequent loading takes ~2-3 secs ...", length=3
            ) as progress_bar:
                try:
                    model = joblib.load(
                        cached_download(hf_hub_url("mikeee/model_s", model_name))
                    )
                except Exception as exc:
                    logger.error(exc)
                    raise
                progress_bar()  # pylint: disable=not-callable
        else:
            try:
                model = joblib.load(
                    cached_download(hf_hub_url("mikeee/model_s", model_name))
                )
            except Exception as exc:
                logger.error(exc)
                raise

    return model
