import logging
import os, time
import mxnet as mx
import numpy as np
import cv2


class MXNetModel:
    def __init__(self, param_path, input_shapes, context=mx.cpu(), label_names=[]):
        self.logger = logging.getLogger(__name__)

        self.logger.info("MXNet model init")
        start_time = time.time()

        self.logger.info('Loading network parameters with prefix: {}'.format(param_path))
        sym, arg_params, aux_params = mx.model.load_checkpoint(param_path, 0)

        self.logger.info('Loading network into MXNet module and binding corresponding parameters')
        self.mod = mx.mod.Module(symbol=sym, label_names=label_names, context=context, logger=self.logger)
        self.mod.bind(for_training=False, data_shapes=input_shapes)
        self.mod.set_params(arg_params, aux_params)

        self.logger.info(
            "MXNet model loaded. Took {:10.2f} ms on PID {}".format((time.time() - start_time) * 1000,
                                                                    str(os.getpid())))

    def infer(self, inference_input):
        self.mod.forward(inference_input)
        output = self.mod.get_outputs()[0].asnumpy()
        output = np.squeeze(output)
        results = [output[0].tolist()]
        return results
