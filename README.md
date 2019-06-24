# Parallelize-ml-inference
Sample code snippets for parallelizing across multiple CPU/GPUs on a single machine to speed up deep learning inference



## Instruction to run the inference script using python multiprocessing
 
On a GPU-enabled machine (e.g. a p3 or p2 EC2 instance), make sure GPU drivers are properly installed along with MXNet:

* Installing Cuda: https://developer.nvidia.com/cuda-downloads 
* Installing MXNet: http://mxnet.incubator.apache.org/versions/master/install/index.html

Alternatively, launch an p3/p2 instance with [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) that have preinstalled CUDA and ML frameworks. 

If you are using a EC2 instance crated from Deep Learning AMI, activate the conda environment: 

```
source activate mxnet_p36
```

Download the source code and model artifacts

```
git clone https://github.com/aws-samples/parallelize-ml-inference.git
cd parallelize-ml-inference/resources/model/
wget https://angelaw-workshop.s3.amazonaws.com/ml/od/model/model.tar.gz
tar -xvzf model.tar.gz
```

Then run the script: 

```
cd ~/parallelize-ml-inference
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python3 src/parallize_inference_pool.py -n 100 -p 2
```
To see the list of supported arguments: 

```
python3 src/parallize_inference_pool.py  -h
usage: parallize_inference_pool.py [-h] [-n NUMBER_OF_INFERENCE]
                                   [-p NUMBER_OF_PROCESS]
                                   [-m MODEL_PARAM_PATH]
                                   [-i INPUT_DIRECTORY_PATH]
                                   [-o OUTPUT_FILE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -n NUMBER_OF_INFERENCE, --number-of-inference NUMBER_OF_INFERENCE
                        number of inferences to make.
  -p NUMBER_OF_PROCESS, --number-of-process NUMBER_OF_PROCESS
                        size of processes pool.
  -m MODEL_PARAM_PATH, --model-param-path MODEL_PARAM_PATH
                        path + prefix to the model parameter files. Default to
                        ./resources/model/deploy_model_algo_1
  -i INPUT_DIRECTORY_PATH, --input-directory-path INPUT_DIRECTORY_PATH
                        path to the input file directory to do inference on.
                        Default is ./resources/imgs/
  -o OUTPUT_FILE_PATH, --output-file-path OUTPUT_FILE_PATH
                        path to the output file. Default is output.json
```



## License Summary

This sample code is made available under the MIT-0 license. See the LICENSE file.
