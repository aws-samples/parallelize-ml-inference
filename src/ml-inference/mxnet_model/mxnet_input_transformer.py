import cv2
import numpy as np
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

import mxnet as mx

def transform_input(filepath, reshape):
    # Switch RGB to BGR format (which ImageNet networks take)
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    if img is None:
        return []

    # Resize image to fit network input
    img = cv2.resize(img, reshape)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    transformed = Batch([mx.nd.array(img)])
    return transformed
