/*
 **************************************************************************
 *
 *
 * NETWORK LAYER DEFINITIONS (for easy copy*paste, NOT used as codebase)
 *
 *
 **************************************************************************
 
 // Sample definition for the most simplest 2-layer feed-forward network
 
 LayerDefinition inputLayer = {
 .layerType       = INPUT,
 .nodeMap         = (Volume){.width=MNIST_IMG_WIDTH, .height=MNIST_IMG_HEIGHT}
 };
 
 LayerDefinition outputLayer = {
 .layerType       = OUTPUT,
 .activationType  = SIGMOID,
 .nodeMap         = (Volume){.width=10}
 };
 
 LayerDefinition *layerDefs = setLayerDefinitions(2, inputLayer, outputLayer);
 optimizeHyperParameters(2, layerDefs);
 
 
 // SIGMOID 0.015   89.04%
 // SIGMOID  0.0125  89.98%  on validation set   89.27% on test set
 // SIGMOID  0.0125  91.02%  on validation set   91.61% on test set   (after 913,000 trainings!!)
 
 // TANH     0.0016  70.70%  on validation set
 // TANH     0.0008  78.56%  on validation set   79.34% on test set   (after 675,000 trainings!!)
 
 
 // RELU     0.0008  87.93%  on validation set
 // RELU     0.0008  89.84%  on validation set   90.68% on test set   (after 1,075,000 trainings!!)
 
 **************************************************************************

// Sample definition for a 3-layer network

LayerDefinition inputLayer = {
    .layerType       = INPUT,
    .nodeMap         = (Volume){.width=MNIST_IMG_WIDTH, .height=MNIST_IMG_HEIGHT}
};

LayerDefinition hiddenLayer = {
    .layerType       = FULLY_CONNECTED,
    .activationType  = SIGMOID,
    .nodeMap         = (Volume){.width=20}  // @attention For FULLY_CONNECTED layers ignore values for height+depth
};

LayerDefinition outputLayer = {
    .layerType       = OUTPUT,
    .activationType  = SIGMOID,
    .nodeMap         = (Volume){.width=10}
};


LayerDefinition *layerDefs = setLayerDefinitions(3, inputLayer, hiddenLayer, outputLayer);

optimizeHyperParameters(3, layerDefs);

 // SIGMOID 20 nodes,  learning rate 0.0817,    91.45%  on validation set     90.40% on testing set
 // SIGMOID 20 nodes   learning rate 0.0790,    90.94%  on validation set     92.98% on testing set (525,000 runs)
 
// 300 nodes, learning rate 0.0700,    93.88%  on validation set  93.08% on test set

 **************************************************************************
 
 // Sample definition of a n-layer convolutional network
 
 LayerDefinition inputLayer = {
 .layerType       = INPUT,
 .nodeMap         = (Volume){.width=MNIST_IMG_WIDTH, .height=MNIST_IMG_HEIGHT}
 };
 
 LayerDefinition hiddenLayer1 = {
 .layerType       = CONVOLUTIONAL,
 .activationType  = RELU,
 .nodeMap         = (Volume){.width=12, .height=12, .depth=5},
 .filter          = 7
 };
 
 LayerDefinition outputLayer = {
 .layerType       = OUTPUT,
 .activationType  = RELU,
 .nodeMap         = (Volume){.width=10}
 };
 
 
 Network *nn = createNetwork(3, inputLayer, hiddenLayer1, outputLayer);
 nn->learningRate    = 0.0004;
 

**************************************************************************

 // Sample definition of a n-layer convolutional network
 
 LayerDefinition inputLayer = {
 .layerType       = INPUT,
 .nodeMap         = (Volume){.width=MNIST_IMG_WIDTH, .height=MNIST_IMG_HEIGHT}
 };
 
 LayerDefinition hiddenLayer1 = {
 .layerType       = CONVOLUTIONAL,
 .activationType  = RELU,
 .nodeMap         = (Volume){.width=12, .height=12, .depth=5},
 .filter          = 7
 };
 
 LayerDefinition outputLayer = {
 .layerType       = OUTPUT,
 .activationType  = RELU,
 .nodeMap         = (Volume){.width=10}
 };
 


**************************************************************************


 // Training the network by processing the TRAINING dataset NUMBER_OF_EPOCH times
 for (int e=0; e<NUMBER_OF_EPOCHS;e++){
 // Train single epoch
 trainNetwork(nn, e);
 // Gradually decrease learning rate after every second epoch
 //        nn->learningRate *= 0.2;
 }
 
 // Testing the during training derived network using the TESTING dataset
 testNetwork(nn);
 
 

*/
