import logging
import multiprocessing
import os, json
import time
import argparse
import pandas as pd
import utils
from itertools import cycle

import mxnet as mx
from mxnet_model.mxnet_model_factory import MXNetModel
from mxnet_model.mxnet_input_transformer import transform_input


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_OF_PROCESS = 3
IMAGE_DIM = 512
INPUT_SHAPE = [('data', (1, 3, IMAGE_DIM, IMAGE_DIM))]

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-of-inference", help="number of inferences to make. ", default=10)
ap.add_argument("-p", "--number-of-process", help="size of processes pool. ", default=1)
ap.add_argument("-m", "--model-param-path",
                help="path + prefix to the model parameter files. Default to ./resources/model/deploy_model_algo_1",
                default="./resources/model/deploy_model_algo_1")
ap.add_argument("-i", "--input-directory-path",
                help="path to the input file directory to do inference on. Default is ./resources/imgs/ ",
                default='./resources/imgs/')
ap.add_argument("-o", "--output-file-path",
                help="path to the output file. Default is output.json ",
                default='output.json')

# This model object contains the loaded ML model.
# Each Python child process will have its own independent copy of the Model object.
# Allowing each child process to do inference using its own copy
model = None


def init_worker(gpus, model_param_path):
    """
    This gets called to initialize the worker process. Here we load the ML model into GPU memory.
    Each worker process will pull an GPU ID from a queue of available IDs (e.g. [0, 1, 2, 3]) to ensure that multiple
    GPUs are consumed evenly.
    """
    global model
    if not gpus.empty():
        gpu_id = gpus.get()
        logger.info("Using GPU {} on pid {}".format(gpu_id, os.getpid()))
        ctx = mx.gpu(gpu_id)
    else:
        logger.info("Using CPU only on pid {}".format(os.getpid()))
        ctx = mx.cpu()
    model = MXNetModel(param_path=model_param_path,
                       input_shapes=INPUT_SHAPE,
                       context=ctx)


def process_input_by_worker_process(input_file_path):
    logger.debug("processing input {} on pid {}".format(input, str(os.getpid())))

    start_time = time.time()
    transformed_input = transform_input(input_file_path, reshape=(IMAGE_DIM, IMAGE_DIM))
    transform_end_time = time.time()
    output = model.infer(transformed_input)
    inference_end_time = time.time()
    return {"file": input_file_path,
            "result": output,
            "transformTime": (transform_end_time - start_time) * 1000,
            "inferenceTime": (inference_end_time - transform_end_time) * 1000,
            }


def run_inference_in_process_pool(model_param_path, input_paths, num_process, output_path):
    # If GPUs are available, create a queue that loops through the GPU IDs.
    # For example, if there are 4 worker processes and 4 GPUs, the queue contains [0, 1, 2, 3]
    # If there are 4 worker processes and 2 GPUs the queue contains [0, 1, 0 ,1]
    gpus = utils.detect_gpus()
    gpu_ids = multiprocessing.Queue()
    if len(gpus) > 0:
        gpu_id_cycle_iterator = cycle(gpus)
        for i in range(num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))

    # Initialize process pool
    process_pool = multiprocessing.Pool(processes=num_process, initializer=init_worker, initargs=(gpu_ids, model_param_path))

    start = time.time()

    # Feed inputs to process pool to do inference
    pool_output = process_pool.map(process_input_by_worker_process, input_paths)

    logger.info('Processed {} images on {} processes for {:10.4f} seconds '.format(len(input_paths), num_process,
                                                                                   time.time() - start))
    write_output_file(pool_output, output_path)

    df = pd.read_json(output_path, lines=True)
    logger.info('Per input timing stats (unit: ms): \n {}'.format(df.describe()))


def write_output_file(output, file_path):
    with open(file_path, 'w') as f:
        for line in output:
            f.write(json.dumps(line))
            f.write('\n')




def generate_input_path_list(input_directory_path, num_inference):
    """
        :return: A list of files from the input directory, repeated enough times to have at least 'num_inference' elements
    """
    input_files = os.listdir(input_directory_path)
    input_paths = []
    for file in input_files:
        input_paths.append(os.path.join(input_directory_path, file))
    repeated_inputs = []
    while len(repeated_inputs) < num_inference:
        repeated_inputs += input_paths
    return repeated_inputs


def main():
    # Parse command line arguments
    args = vars(ap.parse_args())
    num_inference = int(args["number_of_inference"])
    num_process = int(args["number_of_process"])
    model_param_path = args["model_param_path"]
    input_directory_path = args["input_directory_path"]
    output_path = args["output_file_path"]
    logger.info("parameters: \n{}".format(args))

    # Read a list of files from the input directory. Repeat them until we generate enough number of inputs
    repeated_inputs = generate_input_path_list(input_directory_path, num_inference)
    logger.info("will process {} images".format(len(repeated_inputs)))

    run_inference_in_process_pool(model_param_path, repeated_inputs, num_process, output_path)


if __name__ == "__main__":
    main()
