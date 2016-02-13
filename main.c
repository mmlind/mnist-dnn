/**
 * @file main.c
 *
 * @mainpage MNIST Deep Neural Network
 *
 * @brief A deep (multi-layer) mixed-type (convolutional, feed-forward) neural network handling MNIST
 *
 * @details A deep neural network for MNIST image recognition with the following key features:
 * - supports unlimited number of layers, nodes and weights (only restriction is memory)
 * - supports fully connected and convolutional layers
 * - supports following activation functions: SIGMOID, TANH, RELU
 * - light weight architecture with a very small memory footprint
 * - super fast! :-)
 *
 * @see [Deep Neural Network for MNIST Handwriting Recognition](http://mmlind.github.io/Deep_Neural_Network_for_MNIST_Handwriting_Recognition/)
 * @see http://yann.lecun.com/exdb/mnist/
 * @version [Github Project Page](http://github.com/mmlind/mnist-dnn/)
 * @author [Matt Lind](http://mmlind.github.io)
 * @date February 2016
 *
 */




// Include external libraries
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>
#include <locale.h>

// Include project libraries
#include "dnn.h"
#include "util/mnist-utils.h"
#include "util/mnist-stats.h"
#include "util/screen.h"




/**
 * @brief Trains a network on the MNIST training set
 * @details Trains the network by feeding input, calculating and backpropaging the error, updating weights
 * @param nn A pointer to the network
 */

void trainNetwork(Network *nn){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TRAINING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);

        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(nn);

        // Back propagate the error and adjust weights in all layers accordingly
        backPropagateNetwork(nn, lbl);

        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(nn);
        if (classification!=lbl) errCount++;

        // Display progress during training
        displayTrainingProgress(imgCount, errCount);

    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
}




/**
 * @brief Tests an already trained network on the MNIST testing set
 * @details Follows same steps as training process but without backpropagation and updating weights
 * @param nn A pointer to the network
 */

void testNetwork(Network *nn){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);
        
        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(nn);
        
        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(nn);
        if (classification!=lbl) errCount++;
        
        // Display progress during testing
        displayTestingProgress(imgCount, errCount);
    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
}




/**
 * @details Run a demo that creates a network using a sample network design and ouputs result to console
 */

int main(int argc, const char * argv[]) {
    
    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);
    
    // set locale (for outputting formatted numbers/dates/strings)
    setlocale(LC_ALL, "En_US");
    
    // clear screen of terminal window and display title
    clearScreen();
    printf("MNIST-DNN: A deep neural network processing the MNIST handwritten digit images\n\n");

    // Define the network's overall architecture (layers, nodes, activation function, etc.)
    
    // Define how many layers
    int numberOfLayers = 4;

    // Define details of each layer
    LayerDefinition inputLayer = {
        .layerType       = INPUT,
        .nodeMap         = (Volume){.width=MNIST_IMG_WIDTH, .height=MNIST_IMG_HEIGHT}
    };
    
    LayerDefinition hiddenLayer = {
        .layerType       = FULLY_CONNECTED,
        .activationType  = SIGMOID,
        .nodeMap         = (Volume){.width=500}
    };
    LayerDefinition hiddenLayer2 = {
        .layerType       = FULLY_CONNECTED,
        .activationType  = SIGMOID,
        .nodeMap         = (Volume){.width=150}
    };
    LayerDefinition outputLayer = {
        .layerType       = OUTPUT,
        .activationType  = SIGMOID,
        .nodeMap         = (Volume){.width=10}
    };
    
    
/*
    // Define details of each layer
    LayerDefinition inputLayer = {
        .layerType       = INPUT,
        .nodeMap         = (Volume){.width=MNIST_IMG_WIDTH, .height=MNIST_IMG_HEIGHT}
    };
    
    LayerDefinition hiddenLayer = {
        .layerType       = CONVOLUTIONAL,
        .activationType  = RELU,
        .nodeMap         = (Volume){.width=13, .height=13, .depth=5},
        .filter          = 5
    };
    
    LayerDefinition hiddenLayer2 = {
        .layerType       = CONVOLUTIONAL,
        .activationType  = RELU,
        .nodeMap         = (Volume){.width=6, .height=6, .depth=5},
        .filter          = 3
    };
    
    LayerDefinition outputLayer = {
        .layerType       = OUTPUT,
        .activationType  = RELU,
        .nodeMap         = (Volume){.width=10}
    };
 */
    
    // Create an array to hold all of the above layer definitions (for easier reference throught the code)
    LayerDefinition *layerDefs = setLayerDefinitions(numberOfLayers, inputLayer, hiddenLayer, hiddenLayer2, outputLayer);
    
    // Display details of the network definition/architecture on the screen
    outputNetworkDefinition(numberOfLayers, layerDefs);
    
    // Create a neural network based on the above definition
    Network *nn = createNetwork(numberOfLayers, layerDefs);
    
    // Define additional hyper-parameters (optional)
    nn->learningRate = 0.0004;
    nn->learningRate = 0.005;
    
    // Train the network
    for (int i=0;i<2;i++){
        trainNetwork(nn);
        printf("\n");
        
    }
    
    // Test the network
    testNetwork(nn);
    
    // Free the manually allocated memory for this network
    free(nn);
    free(layerDefs);

    // Calculate and print the program's total execution time
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);

    printf("\n\n DONE! Total execution time: %.1f sec\n\n",executionTime);
    
    return 0;
}


