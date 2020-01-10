import greengrasssdk
from mxnet_model.mxnet_model_factory import MXNetModel
from mxnet_model.mxnet_input_transformer import transform_input
import logging
import os
import time
import json
import utils
import mxnet as mx

ML_MODEL_BASE_PATH = '/ml/od/'
ML_MODEL_PREFIX = 'deploy_model_algo_1'
ML_MODEL_PATH = os.path.join(ML_MODEL_BASE_PATH, ML_MODEL_PREFIX)
OUTPUT_TOPIC = 'blog/infer/output'

IMAGE_DIM = 512
INPUT_SHAPE = [('data', (1, 3, IMAGE_DIM, IMAGE_DIM))]

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')
model = None


# Load the model at startup
def initialize(param_path=ML_MODEL_PATH):
    global model
    gpus = utils.detect_gpus()
    if len(gpus) > 0:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    model = MXNetModel(param_path=param_path, input_shapes=INPUT_SHAPE, context=ctx)


def lambda_handler(event, context):
    """
    Gets called each time the function gets invoked.
    """
    '''Echo message on /in topic to /out topic'''
    client.publish(
        topic='{}/out'.format(os.environ['CORE_NAME']),
        payload=event
    )

    # response = {
    #     'prediction': prediction,
    #     'timestamp': time.time(),
    #     'filepath': filepath
    # }
    # client.publish(topic='blog/infer/output', payload=json.dumps(response))
    return {}


# If this path exists then this code is running on the greengrass core and has the ML resources it needs to initialize.
if os.path.exists(ML_MODEL_BASE_PATH):
    initialize()
else:
    logging.info('{} does not exist and we cannot initialize this lambda function.'.format(ML_MODEL_BASE_PATH))
