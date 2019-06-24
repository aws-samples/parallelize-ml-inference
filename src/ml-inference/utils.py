import multiprocessing
import logging
import mxnet as mx

logger = logging.getLogger(__name__)


def detect_gpus():
    cpu_count = multiprocessing.cpu_count()
    logger.info("{} CPUs detected".format(cpu_count))

    try:
        gpus = mx.test_utils.list_gpus()
    except:
        gpus = []

    if len(gpus) == 0:
        logger.info("No GPU detected.")
    else:
        logger.info("{} GPUs detected".format(len(gpus)))

    return gpus
