/**
 * @file dnn.h
 * @brief Core network library for a deep neural network
 * @author Matt Lind
 * @date February 2016
 */


#ifndef DNN_HEADER
#define DNN_HEADER

// Include project libraries
#include "util/mnist-utils.h"
#include "util/mnist-stats.h"

#define MAX_CONVOLUTIONAL_FILTER 10     // check mechanism to avoid users defining wrong conv models

typedef struct LayerDefinition LayerDefinition;
typedef struct Vector3D Vector3D;
typedef struct Value2D Value2D;
typedef struct Volume Volume;
typedef struct Network Network;
typedef struct Layer Layer;
typedef struct Column Column;
typedef struct Node Node;
typedef struct Connection Connection;

typedef double Weight;
typedef unsigned long ByteSize;

typedef enum LayerType {EMPTY, INPUT, CONVOLUTIONAL, FULLY_CONNECTED, OUTPUT} LayerType;
typedef enum ActFctType {SIGMOID, TANH, RELU, NONE} ActFctType;




/**
 * @brief Data structure defining a 3-dimensional vector used to define the size of a node map
 */

struct Volume{
    int width;
    int height;
    int depth;
};




/**
 * @brief Data structure allowing users to define the characteristics of a network
 */

struct LayerDefinition{
    LayerType layerType;        // what kind of layer is this (INP,CONV,FC,OUT)
    ActFctType activationType;  // what activation function is applied
    Volume nodeMap;             // what is the width/height/depth of this layer
    int filter;                 // size of the filter window (conv layers only)
};




/**
 * @brief Data structure attached to a node and pointing to another node as well as to a weight
 * @details Every node has 2 types of connections: forward and backward.
 * Backward connections are used during feed forward to locate the normalized output in the previous layer.
 * Forward connections are used during back propagation to locate the partial errors in the following layer.
 */

struct Connection{
    Node *nodePtr;              // pointer to the target node
    Weight *weightPtr;          // pointer to a weight that is applied to this connection
};




/**
 * @brief Variably-sized data structure modeling a neuron with a variable number of connections/weights
 */

struct Node{
    ByteSize size;              // actual byte size of this structure in run-time
    Weight bias;                // value of the bias weight of this node
    double output;              // result of activation function applied to this node
    double errorSum;            // result of error back propagation applied to this node
    int backwardConnCount;      // number of connections to the previous layer
    int forwardConnCount;       // number of connections to the following layer
    Connection connections[];   // array of connections
};




/**
 * @brief Variably-sized data structure modeling a vector of nodes
 */

struct Column{
    ByteSize size;              // actual byte size of this structure in run-time
    int maxConnCountPerNode;    // maximum number of connections per node in this layer
    int nodeCount;              // number of nodes in this column
    Node nodes[];               // array of nodes
};




/**
 * @brief Variably-sized data structure holding a definable number of columns that form a layer
 */

struct Layer{
    int id;                         // index of this layer in the network
    ByteSize size;                  // actual byte size of this structure in run-time
    LayerDefinition *layerDef;      // pointer to the definition of this layer
    Weight *weightsPtr;             // pointer to the weights of this layer
    int columnCount;                // number of columns in this layer
    Column columns[];               // array of columns
};




/**
 * @brief Variably-sized data structure that serves as the over container for a whole network
 */

struct Network{
    ByteSize size;                  // actual byte size of this structure in run-time
    double learningRate;            // factor by which connection weight changes are applied
    int weightCount;                // number of weights in the net's weight block
    Weight *weightsPtr;             // pointer to the start of the network's weights block
    Weight nullWeight;              // memory slot for a weight pointed to by dead connections
    int layerCount;                 // number of layers in the network
    Layer layers[];                 // array of layers (of different sizes)
};




/**
 * @brief Returns the number of nodes in a layer
 * @param layerDef Pointer to a layer definition
 */

int getLayerNodeCount(LayerDefinition *layerDef);




/**
 * @brief Returns the number of backward connections of a NODE (not of a layer)
 * @details For FEED FORWARD (HIDDEN and OUTPUT) layers, full connectivity is assumed
 * (each node links to ALL nodes in the previous layer)
 * @param layerDef Pointer to a layer definition
 */

int getNodeBackwardConnectionCount(LayerDefinition *layerDef);




/**
 * @brief Returns the number of weights for a layer (based on a given layer definition)
 * @param layerDef A pointer to the layer definition
 */

int getLayerWeightCount(LayerDefinition *layerDef);




/**
 * @brief Returns the memory (byte) size of the weights block for a specific layer
 * @details Each layer's number of weights may be different due to a different number of connections
 * For FEED FORWARD (HIDDEN and OUTPUT) layers, full connectivity is assumed, CONV layers e.g. share weights.
 * @param layerDef A pointer to a layer definition
 */

ByteSize getLayerWeightBlockSize(LayerDefinition *layerDef);




/**
 * @brief Returns the memory (byte) size of a specific layer based on a given layer definition
 * @details Each layer's memory size may be different due to a different number of nodes and connections
 * For FEED FORWARD (e.g. OUTPUT) layers, full connectivity is assumed
 * (i.e. each node links to ALL nodes in the previous layer)
 * @param layerDef A pointer to a layer definition
 */

ByteSize getLayerSize(LayerDefinition *layerDef);




/**
 * @brief Calculates the stride (number of nodes/columns that are skipped) in a convolutional kernel
 * @param tgtWidth Number of columns on the x-axis (horizontally) in the TARGET (=previous) layer
 * @param filter Number of columns/nodes on the x-axis in a filter window (@attention ASSSUMES WIDTH=HEIGHT!!)
 * @param srcWidth Number of columns on the x-axis (horizontally) in the SOURCE (=this) layer
 */

int calcStride(int tgtWidth, int filter, int srcWidth);




/**
 * @brief Feeds some Vector data into the INPUT layer of the network
 * @param nn A pointer to the neural network
 * @param v A pointer to the vector holding the input values
 */

void feedInput(Network *nn, Vector *v);




/**
 * @brief Feeds forward (=calculating a node's output value and applying an activation function) layer by layer
 * @details Feeds forward from 2nd=#1 layer (i.e. skips input layer) to output layer
 * @param nn A pointer to the NN
 */

void feedForwardNetwork(Network *nn);




/**
 * @brief Backpropagates the output nodes' errors from output layer backwards to first layer
 *
 * @details The network's backpropagation proceeds in 2 steps:
 *
 * 1. CALCULATE OUTPUT NODES' ERRORS
 * a. Calculate the errorsums in all output cells based on the targetClassification
 *
 * 2. BACKPROPAGATE EACH LAYER
 * a. Update the nodes weights based on actual output and accumulated errorsum
 * b. Calculate the errorsums in all TARGET cells based on errorsum in this layer (calculated in 3)
 *
 * @param nn A pointer to the neural network
 * @param targetClassification The correct/desired classification (=label) of this recognition/image
 */

void backPropagateNetwork(Network *nn, int targetClassification);




/**
 * @brief Returns the network's classification of the input image by choosing the node with the hightest output
 * @param nn A pointer to the neural network
 */

int getNetworkClassification(Network *nn);




/**
 * @brief Creates the neural network based on a given array of layer definitions
 * @details Creates a reserved memory block for this network based on the given layer definitions,
 * and then initializes this memory with the respective layer/node/connection/weights structure.
 * @param layerCount The number of layer definitions inside the layer-definition-array (2nd param)
 * @param layerDefs A pointer to an array of layer definitions
 */

Network *createNetwork(int layerCount, LayerDefinition *layerDefs);




/**
 * @brief Returns a pointer to an array of a variable number of layer definitions
 * @param layerCount Number of layers of the network
 * @param ... Variabe number of layer definition objects
 */

LayerDefinition *setLayerDefinitions(int layerCount, ...);




#endif
