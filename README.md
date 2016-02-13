# MNIST Deep Neural Network
A deep neural network for MNIST image recognition with the following key features:

* supports unlimited number of layers, nodes and weights (only restriction is memory)
* supports fully connected and convolutional layers
* supports following activation functions: SIGMOID, TANH, RELU
* light weight architecture with a very small memory footprint
* __super fast!__ :-)


### Compile and run source code

The repository comes with a pre-configured `makefile`. You can compile the source simply by typing

```
$ make
```

in the project directory. The binary will be created inside the `/bin` folder and can be executed via

```
$ ./bin/mnist-dnn
```

### Code Review

If you're interested in how the code works take a look at my blog entry where I review the code for this deep neueral network in detail.

* Code Review for [Deep Neural Network for MNIST Handwriting Recognition](http://mmlind.github.io/Deep_Neural_Network_for_MNIST_Handwriting_Recognition/)


### Documentation

The  `/doc` folder contains a doxygen configuration file. When you run it with doxygen it will create updated [HTML documentation](https://rawgit.com/mmlind/mnist-3lnn/master/doc/html/index.html) in the `/doc/html` folder.

* Doxygen [Code Documentation](https://rawgit.com/mmlind/mnist-dnn/master/doc/html/)


### MNIST Database

The `/data` folder contains the original MNIST database files.

For more informaton on MNIST see Yann Lecun's [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)



Version 1.0

Published: February 2016
