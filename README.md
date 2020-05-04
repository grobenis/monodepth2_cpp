# Monodepth2.cpp
This is a simple inference implementation for a well-known depth estimation network written in C++. The whole project is based on libtorch API.

## steps to run this simple project

1. download related libraries including OpenCV and [libtorch](https://pytorch.org/get-started/locally/).
2. download my converted torchscript model or convert your trained model into torchscript by yourself. If you don't familiar with torchscript currently, please check the offical [docs](https://pytorch.org/tutorials/advanced/cpp_export.html)
3. prepare a sample image and change its path in main.cpp
4. if you don't have available gpus, please annotate CUDA options in  ```CMakeLists.txt ```

## dockerfile
If you're familiar with docker, you could run this [project](https://hub.docker.com/repository/docker/tengfei2503/maskrcnn-benchmark.cpp) withour the need of installing those libraries. please remember  install nvidia-docker because our projects needs to use gpus.

## Converted models
you could follow ```to_jit.py ``` to create your own torchscript model and use my converted model directly.\
[monodepth2](https://drive.google.com/open?id=1kZ0H_dWjjv07TvmgbENdhPnkuKQEXt9g)\
[packnet-sfm](https://drive.google.com/file/d/14BLOVAMV5ZQeq7tbI1b6GJ9erkSvCszF/view?usp=sharing)
