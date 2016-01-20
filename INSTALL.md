# Overview

Neural Artistic Style relies on these dependencies:

 * [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
 * [cuDNN](https://developer.nvidia.com/cudnn) v3
 * [DeepPy](http://github.com/andersbll/deeppy), Deep learning in Python.
 * [CUDArray](http://github.com/andersbll/cudarray), CUDA-accelerated NumPy.
 * [Pretrained VGG 19 model](http://www.vlfeat.org/matconvnet/pretrained), choose *imagenet-vgg-verydeep-19*.

The detailed installation steps are explained in the sections below.

Note: This installation in mainly for GNU/Linux distributions.

# Neural Artistic Style

1. Download Neural Artistic Style:

    ```
    $ git clone https://github.com/andersbll/neural_artistic_style.git
    ```

# CUDA

Please refer to the [Installation Guides](http://docs.nvidia.com/cuda/index.html#installation-guides) provided by nVidia.

The CUDA toolkit should be installed at `/usr/local/cuda/`.

# cuDNN

1. Download [cuDNN](https://developer.nvidia.com/cudnn) v3.
1. Extract the tarball file to the CUDA directory:

    ```
    $ sudo tar xzf cudnn-7.0-linux-x64-v3.0-prod.tgz -C /usr/loca/cuda
    ```

The tarball file consists of libcudnn static and shared object libraries, and
the library header.

# CUDArray

1. Download CUDArray:

    ```
    $ git clone https://github.com/andersbll/cudarray.git
    ```

1. Build CUDArray:

Before building CUDArray, please make sure Cython>=0.21 has been installed.  If
not, you can install Cython via Pip:

    ```
    $ pip install --user --upgrade cython
    ```

Start to build:

    ```
    # Install shared object library
    $ make
    $ sudo make install  # install into /usr/local/lib by default
    $ echo "export LD_LIBRARY_PATH=\"/usr/local/lib:$LD_LIBRARY_PATH\"" >> $HOME/.bashrc
    $ source $HOME/.bashrc

    # Install Python modules
    $ sudo python setup.py install
    ```

If you get the error messages when executing `make`, that means you might be
using cuDNN v4 instead of v3 (Issue #36):

    ```
    src/nnet/cudnn.cpp:206:5: error: cannot convert ‘const float*’ to ‘cudnnConvolutionBwdFilterAlgo_t’ for argument ‘8’ to ‘cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, void*, size_t, const void*, cudnnFilterDescriptor_t, void*)’
         ));
         ^
    ./include/cudarray/nnet/cudnn.hpp:85:44: note: in definition of macro ‘CUDNN_CHECK’
     #define CUDNN_CHECK(status) { cudnn_check((status), __FILE__, __LINE__); }
 
    compilation terminated due to -Wfatal-errors.
    make: *** [src/nnet/cudnn.o] Error 1
    ```
                                                                                                                                         
https://github.com/andersbll/cudarray/issues/36

# DeepPy

1. Download DeepPy:

    ```
    $ git clone https://github.com/andersbll/deeppy.git
    ```

1. Copy the `deeppy` module direcotry into the Neural Artistic Style directory:

    ```
    $ cp -a deeppy/deeppy neural_artistic_style
    ```

# Pretrained VGG 19 Model

1. Download the pretrained VGG 19 model:

    ```
    $ cd neural_artistic_style
    $ wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
    ```

The model size is around 510 MB.

# Troubleshooting

## Out of Memory Issue

If you get the out of memory error messages when executing
neural_artistic_style.py (Issue #26):

    ```
    Traceback (most recent call last):
      File "neural_artistic_style.py", line 138, in <module>
        run()
      File "neural_artistic_style.py", line 130, in run
        cost = np.mean(net.update())
      File "neural_artistic_style/style_network.py", line 130, in update
        next_x = layer.fprop(next_x)
      File "neural_artistic_style/deeppy/feedforward/convnet_layers.py", line 71, in fprop
        poolout = self.pool_op.fprop(x)
      File "/usr/local/lib/python2.7/dist-packages/cudarray-0.1.dev-py2.7-linux-x86_64.egg/cudarray/nnet/pool.py", line 34, in fprop
        poolout = ca.empty(poolout_shape, dtype=imgs.dtype)
      File "/usr/local/lib/python2.7/dist-packages/cudarray-0.1.dev-py2.7-linux-x86_64.egg/cudarray/cudarray.py", line 246, in empty
        return ndarray(shape, dtype=dtype)
      File "/usr/local/lib/python2.7/dist-packages/cudarray-0.1.dev-py2.7-linux-x86_64.egg/cudarray/cudarray.py", line 36, in __init__
        self._data = ArrayData(self.size, dtype, np_data)
      File "cudarray/wrap/array_data.pyx", line 16, in cudarray.wrap.array_data.ArrayData.__init__ (./cudarray/wrap/array_data.cpp:1401)
      File "cudarray/wrap/cudart.pyx", line 12, in cudarray.wrap.cudart.cudaCheck (./cudarray/wrap/cudart.cpp:763)
    ValueError: out of memory
    ```
                                                                                                                                         
Here are some solutions:

 1. Use GPU with larger memory.
 1. Use smaller input and style images.
 1. Use CPU instead of GPU.  Set CUDARRAY_BACKEND to 'numpy' as workaround.

https://github.com/andersbll/neural_artistic_style/issues/26
