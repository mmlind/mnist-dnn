/**
 * @file dnn.c
 * @brief Core network library for a deep neural network
 * @author Matt Lind
 * @date February 2016
 */


// Include external libraries
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>

// Include project libraries
#include "util/mnist-utils.h"
#include "util/screen.h"
#include "dnn.h"




#define OUT_OF_RANGE -1     // A marker for labeling node ids that are outside the given node map




/**
 * @brief Returns the number of columns in a layer
 * @param layerDef Pointer to a layer definition
 */

int getLayerColumnCount(LayerDefinition *layerDef){
    int columnCount = layerDef->nodeMap.width * layerDef->nodeMap.height;
    return columnCount;
}




/**
 * @brief Returns the number of nodes in a layer
 * @param layerDef Pointer to a layer definition
 */

int getLayerNodeCount(LayerDefinition *layerDef){
    // @attention All 3 dimensions must have been defaulted to 1 if undefined
    int nodeCount = getLayerColumnCount(layerDef) * layerDef->nodeMap.depth;
    return nodeCount;
}




/**
 * @brief Returns the number of backward connections of a NODE (not of a layer)
 * @details For FEED FORWARD (HIDDEN and OUTPUT) layers, full connectivity is assumed
 * (each node links to ALL nodes in the previous layer)
 * @param layerDef Pointer to a layer definition
 */

int getNodeBackwardConnectionCount(LayerDefinition *layerDef){
    
    int connCount = 0;           // default for INPUT LAYER
    
    switch (layerDef->layerType) {
        case INPUT: {
            break;
        }
        case FULLY_CONNECTED:
        case OUTPUT: {
            connCount = getLayerNodeCount(layerDef-1);
            break;
        }
        case CONVOLUTIONAL: {
            connCount = layerDef->filter * layerDef->filter * (layerDef-1)->nodeMap.depth;
            break;
        }
        default:{
            printf("Error! Wrong/Missing layer type definition! ABORT!!\n");
            exit(1);
            break;
        }
    }
    
    return connCount;
}




/**
 * @brief Returns the number of forward connections of a NODE (not of a layer)
 * @attention The number of FORWARD connections in one layer is NOT the same as the number of BACKWARD connections in the following layer!!
 * @param layerDef A pointer to the layer definition
 */

int getNodeForwardConnectionCount(LayerDefinition *layerDef){
    
    // INPUT and OUTPUT layers don't have any forward connections
    if (layerDef->layerType==INPUT || layerDef->layerType==OUTPUT) return 0;
    
    int connCount = 0;
    
    // Check the layerType of the NEXT(!!) layer
    switch ((layerDef+1)->layerType) {
        case OUTPUT:
        case FULLY_CONNECTED:{
            connCount = getLayerNodeCount(layerDef+1);
            break;
        }
        case CONVOLUTIONAL:{
            // If the next layer is a convolutional layer, then the number of
            // forward connections per node is NOT fix but varies
            // i.e. we define here the MAXIMUM conn count of a node
            connCount = pow((layerDef+1)->filter,2) * (layerDef+1)->nodeMap.depth;
            break;
        }
        default:{
            printf("Error! Wrong/Missing layer type definition! ABORT!!\n");
            exit(1);
            break;
        }
    }
    
    return connCount;
}




/**
 * @brief Returns the number of weights for a layer (based on a given layer definition)
 * @param layerDef A pointer to the layer definition
 */

int getLayerWeightCount(LayerDefinition *layerDef){
    
    int weightCount = 0;
    
    switch (layerDef->layerType) {
            
        case INPUT: {
            break;
        }
        case FULLY_CONNECTED:
        case OUTPUT: {
            weightCount = getLayerNodeCount(layerDef) * getLayerNodeCount(layerDef-1);;
            break;
        }
        case CONVOLUTIONAL: {
            weightCount = layerDef->filter * layerDef->filter * layerDef->nodeMap.depth * (layerDef-1)->nodeMap.depth;
            break;
        }
        default:{
            printf("Error! Wrong/Missing layer type definition! ABORT!!\n");
            exit(1);
            break;
        }
    }
    
    return weightCount;
}




/**
 * @brief Returns the number of columns in a layer (based on a give layer definition)
 * @param layerDef A pointer to the layer definition for this layer
 */

int getColumnCount(LayerDefinition *layerDef){
    // @attention nodeMap dimensions must have been defaulted to 1 if undefined
    int colCount = layerDef->nodeMap.width * layerDef->nodeMap.height;
    return colCount;
}




/**
 * @brief Returns the memory (byte) size of the weights block for a specific layer
 * @details Each layer's number of weights may be different due to a different number of connections 
 * For FEED FORWARD (HIDDEN and OUTPUT) layers, full connectivity is assumed, CONV layers e.g. share weights.
 * @param layerDef A pointer to a layer definition
 */

ByteSize getLayerWeightBlockSize(LayerDefinition *layerDef){
    
    int weightCount = getLayerWeightCount(layerDef);
    ByteSize size   = weightCount * sizeof(Weight);
    
    return size;
}




/**
 * @brief Returns the memory size of the network's weights block based on a given array of layer definitions
 * @details Each layer's number of weights may be different due to a different number of nodes & connections 
 * The weight block is a block of memory that is located inside the network object, AFTER the layers.
 * @param layerCount The number of layers in the network
 * @param layerDefs A pointer to an array of layer definitions
 */

ByteSize getNetworkWeightBlockSize(int layerCount, LayerDefinition *layerDefs){
    
    ByteSize size = 0;
    
    for (int l=0; l<layerCount; l++)
        size += getLayerWeightBlockSize(layerDefs+l);
    
    return size;
}




/**
 * @brief Returns the memory (byte) size of a node based on a given layer definition
 * @details Each layer's nodes' memory size may be different due to a different number of connections
 * For FEED FORWARD (HIDDEN and OUTPUT) layers, full connectivity is assumed
 * (each node links to ALL nodes in the previous layer)
 * @param layerDef Pointer to a layer definition
 */

ByteSize getNodeSize(LayerDefinition *layerDef){
    
    int backwardConnectCount = getNodeBackwardConnectionCount(layerDef);
    int forwardConnectCount  = getNodeForwardConnectionCount(layerDef);
    
    int connectCount = backwardConnectCount + forwardConnectCount;
    
    ByteSize nodeSize = sizeof(Node) + (connectCount * sizeof(Connection));
    
    return nodeSize;
}




/**
 * @brief Returns the memory (byte) size of a column based on a given layer definition
 * @param layerDef A pointer to a layer definition
 */

ByteSize getColumnSize(LayerDefinition *layerDef){
    
    int nodesPerColumn = layerDef->nodeMap.depth;
    
    ByteSize nodeSize = getNodeSize(layerDef);
    
    ByteSize columnSize = sizeof(Column) + (nodesPerColumn * nodeSize);
    
    return columnSize;
}




/**
 * @brief Returns the memory (byte) size of a specific layer based on a given layer definition
 * @details Each layer's memory size may be different due to a different number of nodes and connections
 * For FEED FORWARD (e.g. OUTPUT) layers, full connectivity is assumed 
 * (i.e. each node links to ALL nodes in the previous layer)
 * @param layerDef A pointer to a layer definition
 */

ByteSize getLayerSize(LayerDefinition *layerDef){
    
    ByteSize columnSize = getColumnSize(layerDef);
    
    ByteSize layerSize = sizeof(Layer) + (getLayerColumnCount(layerDef) * columnSize);
    
    return layerSize;
}





/**
 * @brief Returns the memory size of the network based on an array of layer definitions and weightBlockSize
 * @param layerCount Number of defined layers for this network
 * @param layerDefs Array of layer definitions
 *
 * number of columns          = width * height
 * number of nodes per column = depth
 * number of nodes            = width * height * depth
 * number of connections      = filter * depth of previous layer * number of nodes
 * number of weights          = filter * depth of previous layer * depth of this layer
 */

ByteSize getNetworkSize(int layerCount, LayerDefinition *layerDefs){

    ByteSize size = sizeof(Network);
    
    for (int i=0; i<layerCount; i++){
        ByteSize lsize =getLayerSize(layerDefs+i);
        size += lsize;
    }
    
    // get size of weight memory block (located within network, after layers)
    ByteSize weightBlockSize = getNetworkWeightBlockSize(layerCount, layerDefs);
    
    // add weight block size to the network
    size += weightBlockSize;
    
    return size;
}




/**
 * @brief Returns a pointer to a specific node defined by its id from a given layer
 * @details The node is retrieved by moving a pointer from this layer's 1st node forward by id*nodeSize 
 * (it is NOT possible to retrieve a node simply via an array because the actual size of a node 
 * depends on its number of connections)
 * @param column A pointer to the column where this node is located in
 * @param nodeId The id of the node that is to be returned
 */

Node *getColumnNode(Column *column, int nodeId) {
    
    ByteSize nodeSize = sizeof(Node) + ( column->maxConnCountPerNode * sizeof(Connection));
    
    uint8_t *sbptr = (uint8_t*) column->nodes;
    
    sbptr += nodeId * nodeSize;
    
    return (Node*) sbptr;
}




/**
 * @brief Returns a pointer to a specific column defined by its id
 * @details The column is retrieved by moving a pointer from the layer's 1st column forward
 * (it is NOT possible to retrieve a column simply via an array because the actual size of each column 
 * depends on its number/sizes of nodes and thus is variable
 * @param layer A pointer to the layer from which to get the column
 * @param columnId The id of the column that is to be retrieved/accessed
 */

Column *getLayerColumn(Layer *layer, int columnId) {
    
    ByteSize columnSize = getColumnSize(layer->layerDef);
    uint8_t *sbptr = (uint8_t*) layer->columns;
    
    sbptr += columnId * columnSize;
    
    return (Column*) sbptr;
}




/**
 * @brief Returns a pointer to a specific node defined by its layer, column and node id
 * @param layer A pointer to a network layer
 * @param columnId The id of the column inside this layer
 * @param nodeId The id of the node inside this column
 */

Node *getNetworkNode(Layer *layer, int columnId, int nodeId) {
    
    Column *column = getLayerColumn(layer, columnId);
    Node *node = getColumnNode(column, nodeId);
    
    return node;
}




/**
 * @brief Returns a pointer to a specific layer defined by its id from the network
 * @details The layer is retrieved by moving a pointer from the network's 1st layer forward by layerId*layerSize
 * (it is NOT possible to retrieve a layer simply via an array because the actual size of EACH layer depends on its number/sizes of nodes)
 * @param nn A pointer to the NN
 * @param layerId The id of the layer that is to be returned
 */

Layer *getNetworkLayer(Network *nn, int layerId){
    
    // Create SBPTR pointing to the start of the layers
    uint8_t *sbptr = (uint8_t*) nn->layers;
    
    // Move pointer forward layer by layer until the LID layer is reached
    for (int i=0;i<layerId;i++){
        Layer *tmpLayer = (Layer*) sbptr;
        sbptr += tmpLayer->size;
    }
    
    return (Layer*) sbptr;
}




/**
 * @brief Returns the result of applying the given outputValue to the derivate of the activation function
 * @param outVal Output value that is to be back propagated
 * @param actType The type of activation function that was applied during feed forward (SIGMOID/TANH/RELU)
 */

Weight getDerivative(Weight outVal, ActFctType actType){

    Weight d = 0;
    
    switch (actType) {
        case SIGMOID:
            d = outVal * (1-outVal);
            break;
            
        case TANH:
            d = 1-pow(tanh(outVal),2);
            break;
            
        case RELU:
            d = 1 / (1 + pow(M_E,-outVal));
            break;
            
        case NONE:
            d = 1;
            break;
            
        default:
            printf("Undefined derivative function! ABORT!\n");
            exit(1);
            break;
    }
    
    return d;
}




/**
 * @brief Updates a node's weights based on given learning rate
 * @details The accumulated error (difference between desired output and actual output) of this node
 * must have been calculated before and attached to the node (= ->errorSum)
 * @param updateNode A pointer to the node whose weights are to be updated
 * @param learningRate The factor with which errors are applied to weights
 */

void updateNodeWeights(Node *updateNode, double learningRate){
    
    // @attention When updating the weights, only use the BACKWARD connections
    for (int i=0; i<updateNode->backwardConnCount; i++){
        
        Node *prevLayerNode = updateNode->connections[i].nodePtr;
        
        if (prevLayerNode!=NULL){
            *updateNode->connections[i].weightPtr += (learningRate * prevLayerNode->output * updateNode->errorSum);
        }
    
    }
    
    // update bias weight
    updateNode->bias += (learningRate * 1 * updateNode->errorSum);
    
}




/**
 * @brief Returns the total error of a node by adding up all the partial errors from the following layer
 * @details To speed up back propagation the partial errors are referenced via the node's forward connections
 * @param thisNode A pointer to the node whose (to be back propagated) error is to be calculated
 */

double calcNodeError(Node *thisNode) {
   
    double nodeErrorSum = 0;

    int forwardConnStart = thisNode->backwardConnCount;
    
    for (int c=0; c<thisNode->forwardConnCount; c++){
        
        Node  *targetNode = thisNode->connections[forwardConnStart + c].nodePtr;
        Weight *weightPtr = thisNode->connections[forwardConnStart + c].weightPtr;

        nodeErrorSum += targetNode->errorSum * *weightPtr;
        
    }

    return nodeErrorSum;
}




/**
 * @brief Back propagates network error to hidden layer
 * @details Backpropagating a layer means looping through all its nodes' connections,
 * and update the errorSum attached to the TARGET node (=previous layer) of each connection
 * i.e. when "backpropagating" layer x, then the errorSum of the nodes of layer x-1 are calculated
 * @param nn A pointer to the neural network
 * @param layerId The id of the layer that is to be back propagated
 */

void backPropagateLayer(Network *nn, int layerId){
    
    Layer *hl = getNetworkLayer(nn, layerId);

    for (int c=0; c<hl->columnCount; c++){
        
        for (int n=0; n<hl->columns[0].nodeCount; n++){
            
            Node *hn = getNetworkNode(hl,c,n);
            
            hn->errorSum = calcNodeError(hn) * getDerivative(hn->output, hl->layerDef->activationType);

            updateNodeWeights(hn, nn->learningRate);
            
        }
        
    }
    
}




/**
 * @brief Calculates the error (difference of desired classification vs actual node output) of each output node
 * and back propagates the error in the output layer to the previous layer
 * @details The error is calculated based on the given target classification (= image label)
 * and is stored in each output node so that it can be backpropagated later
 * @param nn A pointer to the neural network
 * @param targetClassification The correct/desired classification (=label) of this recognition/image
 */

void backPropagateOutputLayer(Network *nn, int targetClassification){
    
    Layer *ol = getNetworkLayer(nn, nn->layerCount-1);
    
    for (int o=0;o<ol->columnCount;o++){
    
        for (int n=0; n<ol->columns[0].nodeCount; n++){
            
            Node *on = getNetworkNode(ol,o,n);
            
            int targetOutput = (o==targetClassification)?1:0;
            
            double errorDelta = targetOutput - on->output;
            
            on->errorSum = errorDelta * getDerivative(on->output, ol->layerDef->activationType);

            updateNodeWeights(on, nn->learningRate);
            
        }
        
    }
    
}




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

void backPropagateNetwork(Network *nn, int targetClassification){

    backPropagateOutputLayer(nn, targetClassification);

    // Loop backwards from the output layer to the SECOND=#1 layer
    // (the FIRST=#0 layer is the input layer)
    for (int i=(nn->layerCount)-2; i>0; i--) backPropagateLayer(nn, i);
    
}




/**
 * @brief Performs an activiation function to a specified node
 * @param node Pointer to the node that is to be "activated"
 * @param actType The type of activation function to be applied (SIGMOID/TANH/RELU)
 */

void activateNode(Node *node, ActFctType actType){
    
    switch (actType) {
        case SIGMOID:
            node->output = 1 / (1 + (exp((Weight)-node->output)) );
            break;
            
        case TANH:
            node->output = tanh(node->output);
            break;
            
        case RELU:
            node->output =   log(1 + pow(M_E,node->output));
            break;
            
        case NONE:
            break;
            
        default:
            printf("Undefined activation function! ABORT!\n");
            exit(1);
            break;
    }
    
    
}




/**
 * @brief Calculates the output value of a specified node 
 * @details Calculates the vector product of a node's weights with the connections' target nodes' outputs
 * @param node Pointer to the node whose output is to be calculated
 */

void calcNodeOutput(Node *node){
    
    // Start by adding the bias
    node->output = node->bias;

    // @attention When calculating node output only loop through the BACKWARD connections
    for (int i=0; i<node->backwardConnCount;i++){
        
        Node *targetNode = node->connections[i].nodePtr;
        
        if (targetNode != NULL) {
            Weight weight = *node->connections[i].weightPtr;
            node->output += targetNode->output * weight;
        }
    
    }
    
}




/**
 * @brief Calculates the output values of all nodes of a given layer
 * @param layer Pointer to the layer whose nodes are to be activated/calculated
 */

void calcNetworkLayer(Layer *layer){

    for (int c=0;c<layer->columnCount; c++){
        
        for (int n=0; n<layer->columns[0].nodeCount; n++){
            
            Node *node = getNetworkNode(layer, c, n);
            
            calcNodeOutput(node);
            activateNode(node, layer->layerDef->activationType);
            
        }
        
    }
}




/**
 * @brief Feeds forward (=calculating a node's output value and applying an activation function) layer by layer
 * @details Feeds forward from 2nd=#1 layer (i.e. skips input layer) to output layer
 * @param nn A pointer to the NN
 */

void feedForwardNetwork(Network *nn){
    
    for (int l=1; l<nn->layerCount; l++){  // @ATTENTION: Skip the first (=INPUT) layer!
        Layer *layer = getNetworkLayer(nn, l);
        calcNetworkLayer(layer);
    }
    
}




/**
 * @brief Feeds some Vector data into the INPUT layer of the network
 * @param nn A pointer to the neural network
 * @param v A pointer to the vector holding the input values
 */

void feedInput(Network *nn, Vector *v) {
    
    Layer *inputLayer = nn->layers;     // @warning Input layer MUST be the FIRST layer in the network
    
    if (v->count != inputLayer->columnCount){
        printf("Number of values in the input vector must be the same as number of nodes in the NN's INPUT layer! ABORT!!\n");
        exit(1);
    }
    
    // Copy the vector content to the "output" field of the input layer nodes
    for (int i=0; i<v->count;i++){
        Node *inputNode = getNetworkNode(inputLayer, i, 0);     // use 1st level of the column
        inputNode->output = v->vals[i];
    }
    
}




/**
 * @brief Returns the network's classification of the input image by choosing the node with the hightest output
 * @param nn A pointer to the neural network
 */

int getNetworkClassification(Network *nn){
    
    // get output layer
    Layer *l = getNetworkLayer(nn, nn->layerCount-1);   // @warning output layer must be defined as LAST layer
    
    Weight maxOut = 0;
    int maxInd = 0;
    
    for (int i=0; i<l->columnCount; i++){
        
        Node *on = getNetworkNode(l,i,0); // only consider/use 1st level of column
    
        if (on->output > maxOut){
            maxOut = on->output;
            maxInd = i;
        }
    }
    
    return maxInd;
}




/*
 * @brief Initialize the network's weights in the weight block with random numbers (-1 to +1)
 * @param nn A pointer to the neural network
 */

void initNetworkWeights(Network *nn){
    
    // Init weights in the weight block
    for (int i=0; i<nn->weightCount; i++){
        Weight *w = &nn->weightsPtr[i];
        *w = 0.4 * (Weight)rand() / RAND_MAX;   // multiplying by a number <0 results in better performance
        if (i%2) *w = -*w;                      // make half of the weights negative (for better performance)
    }
    
    // Init weights in the nodes' bias
    for (int l=0; l<nn->layerCount;l++){
        Layer *layer = getNetworkLayer(nn, l);
        for (int c=0; c<layer->columnCount; c++){
            Column *column = getLayerColumn(layer, c);
            for (int n=0; n<column->nodeCount; n++){
                
                // init bias weight
                Node *node = getColumnNode(column, n);
                node->bias = (Weight)rand()/(RAND_MAX);
                if (n%2) node->bias = -node->bias;  // make half of the bias weights negative
                // alternatively can also use a constant bias, e.g.: node->bias = 0.1;
                
            }
        }
    }
}





/**
 * @brief Calculates the stride (number of nodes/columns that are skipped) in a convolutional kernel
 * @param tgtWidth Number of columns on the x-axis (horizontally) in the TARGET (=previous) layer
 * @param filter Number of columns/nodes on the x-axis in a filter window (@attention ASSSUMES WIDTH=HEIGHT!!)
 * @param srcWidth Number of columns on the x-axis (horizontally) in the SOURCE (=this) layer
 */

int calcStride(int tgtWidth, int filter, int srcWidth){
    
  return ceil(((double)tgtWidth - filter)/(srcWidth-1));
    
}





/**
 * @brief Returns an array of FILTER-many column ids representing a moving x*y kernel window in the target layer
 * @details The node ids are calculated relative to the nodeId of the parent/calling feature map.
 * The size of the moving filter/frame is defined in the parent/calling layer's filter width/height values.
 * The size of the parent/calling feature map is bigger than the target feature map, 
 * i.e. each x'th (2nd) node, horizontally as well as vertically, is skipped.
 * If a filter's target node would be located outside of the target feature map, -1 is assigned as an id.
 * Later all nodes pointing to a -1 node get assigned a weight pointer to the network's "nn->nullWeight",
 * so that they are still dereferencable. (Because NULL pointers would cause an exception/termination.)
 * @param srcLayer A pointer to the convolutional layer (SOURCE) that creates connections to its previous layer
 * @param srcColId The id of the column in the convolutional (SOURCE) layer that creates the connections
 * @param tgtLayer A pointer to the TARGET layer to which the convolutional/source layer connects to
 * @param filterColIds A pointer to the vector which will return the list of filter ids
 */

void calcFilterColumnIds(Layer *srcLayer, int srcColId, Layer *tgtLayer, Vector *filterColIds){
    
    int srcWidth  = srcLayer->layerDef->nodeMap.width;
    int tgtWidth  = tgtLayer->layerDef->nodeMap.width;
    int tgtHeight = tgtLayer->layerDef->nodeMap.height;
    
    int filter = srcLayer->layerDef->filter;
    
    // @attention For now I'm only using WIDTH to calculate STRIDE
    // Hence this assumes images are qudratic (width=height)
    int stride = calcStride(tgtWidth, filter, srcWidth);
    
    int startX = (srcColId % srcWidth) * stride;
    int startY = floor((double)srcColId/srcWidth) * stride;
    
    int id=0;

    for (int y=0; y<filter; y++){
        
        for (int x=0; x<filter; x++){
            
            int colId = ( (startY+y) * tgtWidth) + (startX+x);
            
            // Check whether target columnId is still within the target node range
            // If NOT then assign a dummy ID ("OUT_OF_RANGE") that is later checked for
            if (
                (floor(colId / tgtWidth) > (startY+y)) ||  // filter exceeds nodeMap on the right
                (colId >= tgtWidth * tgtHeight)            // filter exceeds nodeMap on the bottom
                ) colId = OUT_OF_RANGE;
                
            filterColIds->vals[id] = colId;
            
            id++;
            
        }

    }

}




/**
 * @brief Initializes a single convolutional node by setting its connections weights pointers
 * @details Each convolutional node has connections to a filter/kernel window of nodes in the previous layer.
 * @param node A pointer to the convolutional node whose connections are to be set/initialized
 * @param srcLevel The node's level inside the column. Needed to calculate the position of the respective weight.
 * @param srcLayerWeightPtr A pointer to the weight block of this (=convolutional node's) layer
 * @param targetLayer A pointer to the target layer to which this convolutional node shall connect to
 * @param filterColIds A vector of indeces/positions of the target columns/nodes that this node connects to
 * @param nullWeight A pointer to a weight that is used (1) to initialize or (2) to link "dead" connections
 *
 * The following describes the logic/algorithm for calculating the weights' position
 *
 * use WEIGHT MATRIX (of size filter * filter) for LEVEL 1
 * establish connections from source node to all nodes on LEVEL 1 inside the TARGET FILTER using WM1
 * use WEIGHT MATRIX of LEVEL 2
 * establish connections from source node to all nodes on LEVEL 2 inside the TARGET FILTER using WM2
 * same for all levels of TARGET LAYER
 *
 * move source node to next node (= next node/level in SAME COLUMN!)
 *
 * do same as above, again from LEVEL1-n using WM1-n
 *
 * weights pointer position = (srcLevel * tgtDepth * filterSize) + (tgtLevel * filterSizer) +  filterColId
 *
 * move the filter using the SAME WEIGHT MATRIX across the target layer
 * then move 1 level down in the TARGET layer
 */

void initNetworkBackwardConnectionsConvNode(Node *node, int srcLevel, Weight *srcLayerWeightPtr, Layer *targetLayer, Vector *filterColIds, Weight *nullWeight){
    
    int filterSize = filterColIds->count;
    int tgtDepth   = targetLayer->layerDef->nodeMap.depth;
    
    for (int posInsideFilter=0; posInsideFilter<filterSize; posInsideFilter++){
        
        int targetColId = (int)filterColIds->vals[posInsideFilter];
        
        for (int tgtLevel=0; tgtLevel<tgtDepth; tgtLevel++){
            
            Connection *conn = &node->connections[ (tgtLevel*filterSize)+posInsideFilter];
            
            if (targetColId!=OUT_OF_RANGE){
                
                // Calculate the weight's id (=position in the weights block)
                // @warning (need to consistently use the same logic/order
                // (i.e. first go by target feature map, then by parent feature featureMap, then by node)
                
                // SHARE WEIGHTS
                // Position the weight pointer based on srcLevel, tgtDepth, tgtLevel, filterSize and colId
                
                int weightPosition = (srcLevel*(tgtDepth*filterSize)) + (tgtLevel*filterSize) + posInsideFilter;
                
                Weight *tgtWeight = srcLayerWeightPtr + weightPosition;
                
                Node *tgtNode = getNetworkNode(targetLayer, targetColId, tgtLevel);
                
                conn->nodePtr   = tgtNode;
                conn->weightPtr = tgtWeight;
            
            }
    
            else {
            // if filter pixel is out of range of the target nodes then point to NODES if THIS layer
            // this kind of pointer needs to be captured later i.e. should NOT be CALCULATED/ACTIVATED
            conn->nodePtr   = NULL;
            conn->weightPtr = nullWeight;
    
            }

        }

    }
        
}




/**
 * @brief Initializes a node of a normal, fully connected node
 * @details Creates connections with pointers towards all nodes of the previous layer (=fully connected)
 * @attention The node's bias weight is not initialized here but together with the weights
 * @param thisNode A pointer to the node whose connections are to be added/initialized
 * @param prevLayer A pointer to the PREVIOUS layer which this node will connect to
 * @param nodeWeightPtr A pointer to the memory block that is to store the weights of this node
 */

void initNetworkBackwardConnectionsFCNode(Node *thisNode, Layer *prevLayer, Weight *nodeWeightPtr){
    
    // Actually the counters for connId and nodeWeightId are the same (because layer is fully connected)
    // but I still use 2 different variables to express the different meanings
    int connId = 0;
    int nodeWeightId = 0;
    
    ByteSize columnSize = getColumnSize(prevLayer->layerDef);
    
    uint8_t *sbptr_column = (uint8_t*) prevLayer->columns;
    
    // loop through the columns of the previous layer
    for (int col=0; col<prevLayer->columnCount;col++){
        
        Column *column = (Column *)sbptr_column;
        
        uint8_t *sbptr_node = (uint8_t*) column->nodes;

        // loop through the nodes of the column
        for (int n=0; n<column->nodeCount; n++){

            // @attention Only the backwardConnections are set here. Forward connections need to be set elsewhere.
            Connection *conn = &thisNode->connections[connId];
            
            // Create a pointer to the next available weight in the network's memory block behind the layers
            conn->weightPtr = nodeWeightPtr + nodeWeightId;
            
            // Set target node pointer
            conn->nodePtr = (Node *)sbptr_node;
            
            sbptr_node += getNodeSize(prevLayer->layerDef);

            connId++;
            nodeWeightId++;
            
        }
        
        sbptr_column += columnSize;
        
    }
    
}




/*
 * @brief Initializes the forward connections of a given node
 * @details Loops through all backward connections of the following layer to find out all nodes
 * that link back to this node. Then creates forward connections from this node to those nodes in
 * the next layer and points to the same weight. Forward connections are strictly speaking not required
 * but useful for increasing performance during back propagation.
 * @param thisNode A pointer to the node whose forward connections are to be initialized
 * @param nextLayer A pointer to the following layer to which this node's forward connections point to
 */

void initNetworkForwardConnectionsAnyNode(Node *thisNode, Layer *nextLayer) {
    
    int maxForwardConnCount = thisNode->forwardConnCount;
    
    int forwardConnStart = thisNode->backwardConnCount;
    
    int forwardConnCount = 0;
    
    for (int o=0;o<nextLayer->columnCount;o++){
        
        for (int n=0; n<nextLayer->columns[0].nodeCount; n++){
            
            Node *nextLayerNode = getNetworkNode(nextLayer,o, n);
            
            for (int c=0;c<nextLayerNode->backwardConnCount;c++){
                
                // If the connection of the node in the next layer points back to this node
                // then store this nextNode as a target in the forwardConnections of this node
                // and point the connection to the same weight
                if (nextLayerNode->connections[c].nodePtr == thisNode) {
                    thisNode->connections[forwardConnStart + forwardConnCount].nodePtr = nextLayerNode;
                    thisNode->connections[forwardConnStart + forwardConnCount].weightPtr = nextLayerNode->connections[c].weightPtr;
                    forwardConnCount++;
                    
                    if (forwardConnCount>maxForwardConnCount) {
                        printf("Maximum forward connections exceeded! ABORT!\n\n");
                        exit(1);
                    }
                    
                }
                
            }
            
        }
        
        
    }
    
    thisNode->forwardConnCount = forwardConnCount;
    
}




/*
 * @brief Initializes the forward connections in a layer
 * @param nn A pointer to the neural network
 * @param nn The id of the layer to be initialized
 */

void initNetworkForwardConnections(Network *nn, int layerId) {
    
    // Skip the first=INPUT and the last=OUTPUT layer because they don't have forward connections
    if (layerId==0 || layerId==nn->layerCount-1) return;
    
    Layer *thisLayer = getNetworkLayer(nn, layerId);
    Layer *nextLayer = getNetworkLayer(nn, layerId+1);
    
    for (int c=0; c<thisLayer->columnCount; c++){
        
        Column *column = getLayerColumn(thisLayer, c);
        
        for (int n=0; n<column->nodeCount; n++){
            Node *node = getColumnNode(column, n);
            initNetworkForwardConnectionsAnyNode(node, nextLayer);
            
        }
        
    }
    
}




/**
 * @brief Creates and returns an array of FILTER-many column ids representing a moving x*y kernel window
 * @details The actual calculation of the target column ids takes place in a subfunction. 
 * @param thisLayer A pointer to the convolutional layer (SOURCE) that creates connections to its previous layer
 * @param columnId The id of the column in the source layer
 * @param prevLayer A pointer to the PREVIOUS=TARGET layer to which the convolutional/source layer connects to
 */

Vector *createFilterColumnIds(Layer *thisLayer, int columnId, Layer *prevLayer){
    
    // create an empty vector for storing the target node ids of the filter window
    // number of values in the vector equals the size of the filter window
    // @attention This is done even for non-convolutional layer because their filter is 0 thus no impact
    int colIdCount = thisLayer->layerDef->filter * thisLayer->layerDef->filter;
    ByteSize vectorSize = sizeof(Vector) + (colIdCount * sizeof(double));  // TODO don't need "double" here
    
    // Calculate a matrix of column/node ids depicting a moving filter/kernel window in the target layer
    Vector *filterColIds = (Vector*)malloc(vectorSize);
    filterColIds->count = colIdCount;
    calcFilterColumnIds(thisLayer, columnId, prevLayer, filterColIds);
    
    return filterColIds;
}




/**
 * @brief Sets default values for a node during its initialization
 * @param thisLayer A pointer to the layer in which the node is located
 * @param column A pointer to the column in which the node is located
 * @param node A pointer to the node whose values are to be (re)set
 * @param nullWeight A pointer to the network's null weight
 */

void setNetworkNodeDefaults(Layer *thisLayer, Column *column, Node *node, Weight *nullWeight){
    
    ByteSize nodeSize   = getNodeSize(thisLayer->layerDef);
    
    // Set default values of a node
    node->size     = nodeSize;
    node->bias     = 0;
    node->output   = 0;
    node->errorSum = 0;
    node->backwardConnCount= getNodeBackwardConnectionCount(thisLayer->layerDef);
    node->forwardConnCount = getNodeForwardConnectionCount(thisLayer->layerDef);
    
    // Resest ALL (backward + forward) connections of a node to avoid any undefined pointers
    for (int c=0; c<column->maxConnCountPerNode; c++){
        Connection *conn = node->connections + c;
        conn->nodePtr   = NULL;
        conn->weightPtr = nullWeight;
    }
    
}




/**
 * @brief Initializes the nodes in a given network column
 * @details Creates the column/node structure inside the network's respective memory block.
 * Connections will be initialized with a NULL pointer and a default ->nullWeight pointer
 * (for null weights I don't use NULL to avoid exceptions if it is (mistakenly?) dereferenced)
 * @param nn A pointer to the network
 * @param layerId The index of the layer whose column=nodes are to be initialized
 * @param columnId The index of the column whose nodes are to be initialized
 */

void initNetworkNodes(Network *nn, int layerId, int columnId){
    
    Layer *thisLayer    = getNetworkLayer(nn, layerId);
    Layer *prevLayer    = getNetworkLayer(nn, layerId-1);
    Column *column      = getLayerColumn(thisLayer, columnId);
    ByteSize nodeSize   = getNodeSize(thisLayer->layerDef);
    
    uint8_t *sbptr = (uint8_t*) column->nodes;

    // Create a vector containing the ids of the target columns (conv layers only)
    Vector *filterColIds = createFilterColumnIds(thisLayer, columnId, prevLayer);
    
    // Init all nodes attached to this column
    for (int n=0; n<column->nodeCount; n++){
    
        // Set pointer to the respective node position (using a single byte pointer)
        Node *node = (Node*) sbptr;
        sbptr += nodeSize;

        // Reset node's defaults
        setNetworkNodeDefaults(thisLayer, column, node, &nn->nullWeight);
        
        // Initialize backward connections of fully-connected layer node
        if (thisLayer->layerDef->layerType==FULLY_CONNECTED || thisLayer->layerDef->layerType==OUTPUT){
            
            int nodeId = (columnId * column->nodeCount) + n;

            // @attention When calculating the weightsId, only consider backwardConnections
            int layerWeightsId = nodeId * getNodeBackwardConnectionCount(thisLayer->layerDef);
            Weight *nodeWeight = thisLayer->weightsPtr + layerWeightsId;
            
            initNetworkBackwardConnectionsFCNode(node, prevLayer, nodeWeight);
        }
        
        // Initialize backward conections of convolutional layer node
        if (thisLayer->layerDef->layerType==CONVOLUTIONAL){
            // @attention Nodes on the same level share the same weight block
            initNetworkBackwardConnectionsConvNode(node, n, thisLayer->weightsPtr, prevLayer, filterColIds, &nn->nullWeight);
        }
    
    }

    free(filterColIds);
}




/**
 * @brief Initializes a network layer's column/node structure and sets detault values.
 * @details A column is a vector of nodes. The number of nodes in a colum is defined as the "DEPTH"
 * (or number of feature maps) For non-convolutional layers the DEPTH (i.e. number of nodes in a column) is 1.
 * @param nn A pointer the network
 * @param layerId The id of the layer whose column are to be initialized
 */

void initNetworkColumns(Network *nn, int layerId){
    
    Layer *layer = getNetworkLayer(nn, layerId);
    
    int backwardConnCount = getNodeBackwardConnectionCount(layer->layerDef);
    int forwardConnCount  = getNodeForwardConnectionCount(layer->layerDef);
    
    ByteSize columnSize = getColumnSize(layer->layerDef);
    
    // Init all columns attached to this layer
    for (int c=0; c<layer->columnCount; c++){
        
        // Set pointer to the respective column position (using a single byte pointer)
        uint8_t *sbptr = (uint8_t*) layer->columns;
        sbptr += c * columnSize;
        
        Column *column = (Column*) sbptr;
        
        // Set default values of a node
        column->size     = columnSize;
        column->nodeCount= layer->layerDef->nodeMap.depth;
        column->maxConnCountPerNode = backwardConnCount+forwardConnCount;
        
        // Built-in cross checking to confirm initialization is progressing correctly
        Column *testColumn = getLayerColumn(layer, c);
        if (testColumn != column) {
            printf("Error during column initialization! ABORT!\n");
            exit(1);
        }
        
        // Initialize all nodes of a column
        initNetworkNodes(nn, layerId, c);
        
    }
    
}




/**
 * @brief Initializes a network layer by creating the column/node structure and sets detault values.
 * @param nn A pointer to the neural network
 * @param layerId The id of the layer which is to be initialized
 * @param layerDefs Pointer to an array of layer definitions
 */

void initNetworkLayer(Network *nn, int layerId, LayerDefinition *layerDefs){
    
    LayerDefinition *layerDef = layerDefs + layerId;
    
    // Calculate the layer's position by moving a single byte pointer forward
    // by the total sizes of all previous layers
    uint8_t *sbptr1 = (uint8_t*) nn->layers;
    for (int l=0; l<layerId; l++) sbptr1 += getLayerSize(layerDefs+l);
    Layer *layer = (Layer*) sbptr1;
    
    // Calculate the position of this layer's weights block
    uint8_t *sbptr2 = (uint8_t*) nn->weightsPtr;
    for (int l=0; l<layerId; l++) sbptr2 += getLayerWeightBlockSize(layerDefs+l);
    Weight *w = (Weight*) sbptr2;
    
    // Set default values for this layer
    layer->id              = layerId;
    layer->layerDef        = layerDef;
    layer->weightsPtr      = w;
    layer->size            = getLayerSize(layerDef);
    layer->columnCount     = getColumnCount(layerDef);
    
    // Built-in cross checking to confirm initialization is progressing correctly
    Layer *testLayer = getNetworkLayer(nn, layerId);
    if (testLayer != layer) {
        printf("Error during layer initialization! ABORT!\n");
        exit(1);
    }
    
    // Initialize all columns inside this layer
    initNetworkColumns(nn, layerId);
    
}




/*
 * @brief Initialize the network
 * @details Creates the structure of layers/columns/nodes/connections/ weights inside the net's memory block.
 * Most importantly, it sets each node's connections and pointers to their target node weights.
 * @param nn A pointer to the neural network
 * @param layerCount The number of layers of this network
 * @param layerDefs A pointer to an array of the layer definitions for this network
 */
void initNetwork(Network *nn, int layerCount, LayerDefinition *layerDefs){
    
    // Init the network's layers including their backward connections
    // Backward connections point to target nodes in the PREVIOUS layer and are used during FEED FORWARD
    // (i.e. during calculating node outputs = node activation)
    for (int l=0; l<layerCount; l++) initNetworkLayer(nn, l, layerDefs);
    
    // Init the network's forward connections
    // Forward connections point to target nodes in the FOLLOWING layer that point back to this node, and
    // and are used during BACK PROPAGATION (to speed-up calculating the proportional error)
    // @attention This must be done AFTER(!) the above layer initialization because each layer needs its following layer to have been initialized already!
    for (int l=0; l<layerCount; l++) initNetworkForwardConnections(nn, l);
    
}




/**
 * @brief Sets the network's default values for size, layerCount, 
 * @details Theses values are needed to "navigate" inside the network object
 * @param nn A pointer to the neural network
 * @param layerCount The number of layers of this network
 * @param layerDefs A pointer to an array of the layer definitions for this network
 * @param netSize Total memory size of the network
 */

void setNetworkDefaults(Network *nn, int layerCount, LayerDefinition *layerDefs, ByteSize netSize){
    
    // get size of weight memory block (located within network, after layers)
    ByteSize weightBlockSize = getNetworkWeightBlockSize(layerCount, layerDefs);
    
    // Calculate the exact position of the weightBlock and create a pointer pointing to it
    uint8_t *sbptr = (uint8_t*) nn;
    sbptr += netSize - weightBlockSize;
    
    // Set the network's default values
    nn->size         = netSize;
    nn->layerCount   = layerCount;
    nn->weightsPtr   = (Weight*)sbptr;
    nn->nullWeight   = 0;
    nn->learningRate = 0.001;      // @attention This value should be chosen based on the activation fct.
    
    // Calculate the network's number of weights by adding up the layers
    nn->weightCount = 0;
    for (int l=0; l<layerCount; l++) nn->weightCount += getLayerWeightCount(layerDefs+l);
    
    // Cross-check the network's weight count ("just to make sure :-)")
    if (nn->weightCount != (double)weightBlockSize/sizeof(Weight)) {
        printf("Incorrect weight count! ABORT!");
        exit (1);
    }
    
}




/**
 * @brief Creates the neural network based on a given array of layer definitions
 * @details Creates a reserved memory block for this network based on the given layer definitions,
 * and then initializes this memory with the respective layer/node/connection/weights structure.
 * @param layerCount The number of layer definitions inside the layer-definition-array (2nd param)
 * @param layerDefs A pointer to an array of layer definitions
 */

Network *createNetwork(int layerCount, LayerDefinition *layerDefs){
    
    // Calculate network size
    ByteSize netSize = getNetworkSize(layerCount, layerDefs);
    
    // Allocate memory block for the network
    Network *nn = (Network*)malloc(netSize);
    
    // Set network's default values
    setNetworkDefaults(nn, layerCount, layerDefs, netSize);

    // Output message to inform user in case the initialization process takes longer (large network)
    printf("Initializing network... \n\n");
    
    // Initialize the network's layers, nodes, connections and weights
    initNetwork(nn, layerCount, layerDefs);
    
    // Init all weights -- located in the network's weights block after the last layer
    initNetworkWeights(nn);
    
    return nn;
}




/**
 * @brief Validates the network definition based on a number of rules and best practices
 * @details Checks whether the provided layer definitions define a proper/feasible a neural network
 * @param layerCount Number of defined layers for this network
 * @param layerDefs A pointer to an array of layer definitions
 */

bool isValidNetworkDefinition(int layerCount, LayerDefinition *layerDefs){
    
    bool isValid = true;
    
    // 1st layer must be input layer
    if (layerDefs->layerType!=INPUT) isValid = false;
    
    // last layer must be output layer
    if ((layerDefs+layerCount-1)->layerType!=OUTPUT) isValid = false;
    
    // Apply a number of checks for required fields in each layer definiton
    for (int i=0; i<layerCount; i++){
        
        LayerDefinition *layerDef = layerDefs+i;
        
        // If layer does not meet validation rules, then return false
        if (!isValid) return isValid;
        
        // Each layer must have a predefined TYPE
        if (layerDef->layerType!=INPUT &&
            layerDef->layerType!=CONVOLUTIONAL &&
            layerDef->layerType!=FULLY_CONNECTED &&
            layerDef->layerType!=OUTPUT) isValid=false;
        
        // All layers must have some defined number of nodes
        if (layerDef->nodeMap.width==0 && layerDef->nodeMap.height==0 && layerDef->nodeMap.depth==0) isValid = false;
        
        // Check for minimum and maximum nodeMap definitions per layerType
        // Non-convolutional layers cannot have a DEPTH
        if ((layerDef->layerType == INPUT || layerDef->layerType == FULLY_CONNECTED || layerDef->layerType == OUTPUT ) &&
            (layerDef->nodeMap.depth!=0))
            isValid = false;
        
        // CONVOLUTIONAL layers must be 3-dimensional
        if ((layerDef->layerType == CONVOLUTIONAL) &&
            (layerDef->nodeMap.height==0 || layerDef->nodeMap.depth==0))
            isValid = false;
        
        // CONVOLUTIONAL layers must have a FILTER and a STRIDE
        if (layerDef->layerType==CONVOLUTIONAL && (layerDef->filter == 0)) isValid = false;
        
        // For CONVOLUTIONAL layers the FILTER size must be smaller than the nodeMap of the previous layer
        if (layerDef->layerType==CONVOLUTIONAL) {
            if ((layerDef->filter >= (layerDef-1)->nodeMap.width ) ||
                (layerDef->filter >= (layerDef-1)->nodeMap.height)) isValid = false;
        }
        
        // For CONVOLUTIONAL layers check for maximum FILTER and STRIDE sizes
        if (layerDef->layerType==CONVOLUTIONAL) {
            if ((layerDef->filter > MAX_CONVOLUTIONAL_FILTER )) isValid = false;
        }
        
        // All layers (except the INPUT layer) must have an activationFunction defined
        if (layerDef->layerType!=INPUT) {
            if (layerDef->activationType!=SIGMOID &&
                layerDef->activationType!=TANH    &&
                layerDef->activationType!=RELU    &&
                layerDef->activationType!=NONE)         // @attention "NONE" is for testing pooling layers
                isValid = false;
        }
        
        
        // Move pointer forward to the next layer definition
        layerDef++;
    }
    
    return isValid;
}




/**
 * @brief Returns a pointer to an array of a variable number of layer definitions
 * @param layerCount Number of layers of the network
 * @param layerDefs Variabe number of layer definition objects
 */

void setLayerDefinitionDefaults(int layerCount, LayerDefinition *layerDefs){
    
    for (int l=0; l<layerCount; l++){
        
        // Each dimension of the nodeMap shall always be at least 1, so that the same calculation
        // for nodeCount and weightCount can be used for convolutional and non-convolutional layers
        if ((layerDefs+l)->nodeMap.width ==0) (layerDefs+l)->nodeMap.width =1;
        if ((layerDefs+l)->nodeMap.height==0) (layerDefs+l)->nodeMap.height=1;
        if ((layerDefs+l)->nodeMap.depth ==0) (layerDefs+l)->nodeMap.depth =1;
        
        // Set filter default to 0 so that the filter calculation does not need to check for the layerType
        // (i.e. same calculation used for non-convolutional layers because filter=0 means no calculation)
        if ((layerDefs+l)->layerType!=CONVOLUTIONAL) (layerDefs+l)->filter=0;
        
    }
    
}




/**
 * @brief Returns a pointer to an array of a variable number of layer definitions
 * @param layerCount Number of layers of the network
 * @param ... Variabe number of layer definition objects
 */

LayerDefinition *setLayerDefinitions(int layerCount, ...){
    
    // Reserve memory for dynamically-sized LayerDefinitions array
    LayerDefinition *layerDefs = (LayerDefinition*)malloc(sizeof(LayerDefinition) * layerCount);
    
    // Get layer definitions from function arguments and set all undefined elements to 0/NULL
    va_list fctArgs;
    va_start(fctArgs, layerCount);
    for (int i=0; i<layerCount; i++) layerDefs[i] = va_arg(fctArgs, LayerDefinition);
    va_end(fctArgs);
    
    
    // Validate layer definitions
    bool isValid = isValidNetworkDefinition(layerCount, layerDefs);
    if (!isValid) {
        printf("Invalid Network Definition! ABORT!!\n");
        exit(1);
    }
    
    // Set default values for the layer definitions
    setLayerDefinitionDefaults(layerCount, layerDefs);
    
    return layerDefs;
}
