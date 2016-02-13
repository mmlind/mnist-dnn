//
//  swarm.c
//  mnist-cvnn
//
//  Created by Matt Lind on 2/9/16.
//  Copyright © 2016 Matt Lind. All rights reserved.
//



/*

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "swarm.h"
#include "mlnn.h"
#include "../util/mnist-utils.h"
#include "../util/screen.h"






int getIndexOfMinimum(double *accuracies, double *lrates, int size){
    
    double minVal = 1; // all array values must be smaller than this
    int minInd = 0;
    
    for (int i=0; i<size; i++){
        
        if (accuracies[i] < minVal) {
            minVal = accuracies[i];
            minInd = i;
        }
        
    }
    
    return minInd;
}

int getIndexOfMaximum(double *accuracies, double *lrates, int size){
    
    double maxVal = -1; // all array values must be smaller than this
    int maxInd = 0;
    
    for (int i=0; i<size; i++){
        
        if (accuracies[i] > maxVal) {
            maxVal = accuracies[i];
            maxInd = i;
        }
        
    }
    
    return maxInd;
}



int updateLowestValue(double *accuracies, double *lrates, int size){
    
    int minInd = getIndexOfMinimum(accuracies, lrates, size);
    int maxInd = getIndexOfMaximum(accuracies, lrates, size);
    
    // set new rate as in the middle of max + secondMax
    lrates[minInd] = (lrates[maxInd] + lrates[minInd])/(double)2;
    
    return minInd;
}



void optimizeHyperParameters(int layerCount, LayerDefinition *layerDefs){
    
    //    double swarmLearningRates[NUMBER_OF_SWARM_NETS] = {0.0001,0.0008, 0.0009, 0.0010, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007};
    
    double swarmLearningRates[NUMBER_OF_SWARM_NETS] = {0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0031, 0.0064, 0.0125, 0.0250, 0.0500};
    
    double swarmClassResults[NUMBER_OF_SWARM_NETS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double swarmTrainResults[NUMBER_OF_SWARM_NETS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    
    // Open MNIST Files
    FILE *imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    FILE *labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    // Define range of images to be read
    int fromImageId = 0;
    int toImageId = 5000;   // batchSize
    
    
    // Create 10 swarm nets
    Network *swarmNets[NUMBER_OF_SWARM_NETS];
    for (int n=0; n<NUMBER_OF_SWARM_NETS; n++){
        swarmNets[n] = createNetwork(layerCount, layerDefs);
        //        initNetworkWeights(swarmNets[n]);
    }
    
    // Loop every test run
    int numberOfTestRuns = 9;
    for (int testRun=0; testRun<numberOfTestRuns; testRun++){
        
        // Loop through all swarm nets
        for (int n=0; n<NUMBER_OF_SWARM_NETS; n++){
            
            swarmNets[n]->learningRate = swarmLearningRates[n];
            
            locateCursor(20+n, 1);
            printf("Run %2d  Net %2d  Learning Rate: %8.6f  Training on image...\n",testRun+1, n+1, swarmLearningRates[n]);
            
            trainNetwork(swarmNets[n], n, fromImageId, toImageId, imageFile, labelFile);
            
            printf("Validating on image...\n");
            
            swarmClassResults[n] = validateNetwork(swarmNets[n], n, toImageId, imageFile, labelFile);
            printf("Accuracy: %5.2f%% \n\n",swarmClassResults[n]*100);
            
            
            
        }
        
        
        int minInd = updateLowestValue(swarmClassResults, swarmLearningRates, NUMBER_OF_SWARM_NETS);
        
        // retrain the new net
        trainNetwork(swarmNets[minInd], minInd, 0, toImageId, imageFile, labelFile);
        
        fromImageId = toImageId;
        toImageId *= 1.35;
        
        
    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
    
    int maxInd = getIndexOfMaximum(swarmClassResults, swarmLearningRates, NUMBER_OF_SWARM_NETS);
    
    
    locateCursor(31,1);
    
    printf("Optimization completed. Most effective learning rate is %6.4f achieving %5.2f%% accuracy on the validation set. \n\n",swarmLearningRates[maxInd],swarmClassResults[maxInd]*100);
    
    
    //    testNetwork(swarmNets[maxInd], 33);
    
    
}





//
// @brief Test the trained network by processing the MNIST testing set WITHOUT updating weights (single epoch)
// @param nn A pointer to the NN
//

void testNetwork(Network *nn, int yPos){
    
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
        displayTestingProgress(imgCount, errCount, yPos,1);
        //        displayImage(&img, lbl, classification, 15,6);
        
    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
}




void outputTrainingStatsHeader(){
    
    printf("\nSwarming: Running network agents to find the most effective hyper-parameters (learning rate):\n\n");
    
    printf("Agent   Runs#   LearnRate   TrainSets#   Accuracy%%   Validation%%   Reason for termination\n");
    
}

void outputTrainingStats(SwarmAgent *sa){
    
    moveCursorTo( 1);    printf("%3d"  ,sa->id);
    moveCursorTo(10);    printf("%3d"  ,sa->runCount);
    moveCursorTo(18);    printf("%7.5f",sa->nn->learningRate);
    moveCursorTo(29);    printf("%'9d" ,sa->trainCount);
    moveCursorTo(44);    printf("%6.2f",sa->currAccuracy*100);
    moveCursorTo(58);    printf("%6.2f",sa->validationAccuracy*100);
    
    printf("   ");
}




void validateAgent(SwarmAgent *sa){
    
    int errorCount = 0;
    sa->validationAccuracy = 0;
    
    int fromImageId = 50000;
    int toImageId = 55000;
    fseek(sa->imageFile, sizeof(MNIST_ImageFileHeader), SEEK_SET);

    // Loop through all images in the file
    for (int imgId=fromImageId; imgId<toImageId; imgId++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImageByPosition(sa->imageFile, imgId);
        MNIST_Label lbl = getLabelByPosition(sa->labelFile, imgId);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(sa->nn, inpVector);
        
        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(sa->nn);
        
        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(sa->nn);
        if (classification!=lbl) errorCount++;
        
        sa->validationAccuracy = 1-(errorCount/(double)(imgId-fromImageId));
        
        outputTrainingStats(sa);
        
    }
    // update personal best (need to do this after a minimum batch, otherwise 1 correct guess means accuracy 100%
    if (sa->validationAccuracy > sa->bestAccuracy) sa->bestAccuracy = sa->validationAccuracy;
    
}






void trainAgent(SwarmAgent *sa, int fromImageId, int toImageId){
    
    if (fromImageId>=50000) {
        // reset file to beginning
        fromImageId = 0;
        toImageId -= 50000;
        fseek(sa->imageFile, sizeof(MNIST_ImageFileHeader), SEEK_SET);
    }
    
    if (toImageId>50000) toImageId=50000;
    
    // Loop through all images in the file
    for (int imgId=fromImageId; imgId<toImageId; imgId++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImageByPosition(sa->imageFile, imgId);
        MNIST_Label lbl = getLabelByPosition(sa->labelFile, imgId);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(sa->nn, inpVector);
        
        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(sa->nn);
        
        // Back propagate the error and adjust weights in all layers accordingly
        backPropagateNetwork(sa->nn, lbl);
        
        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(sa->nn);
        if (classification!=lbl) sa->errorCount++;
        
        sa->trainCount++;
   
        sa->currAccuracy = 1- (sa->errorCount/(double)sa->trainCount);
        
        outputTrainingStats(sa);
        
    }
    
}







void startAgent(SwarmAgent *sa){
    
    int     initialBatchSize    = 5000;        // new agents are always trained on x images before any evaluation

    sa->runCount++;
    sa->trainCount = 0;
    sa->errorCount = 0;
    sa->currAccuracy = 0;
    sa->bestAccuracy = 0;
    sa->validationAccuracy = 0;
    
    // Initialize network
    initNetworkWeights(sa->nn);
    
    // Run initial batch
    trainAgent(sa, 0, initialBatchSize);
    
}



void retrainAgent(SwarmAgent *sa){
    
    int factor = 1;
    
    if (sa->runCount % 2 ==0) factor = -1;
    
    int intervalSize = ceil(sa->runCount/(double)2);
    
    double  incrementalChange   = 0.001;
    
    printf("\n");
    
    // Set entry point
    sa->nn->learningRate = sa->startRate + (intervalSize * incrementalChange * factor);
    
    startAgent(sa);
    
}


void evalAgent(SwarmAgent *sa){
    
    double targetAccuracy      = 0.95;
    double minAccuracy         = 0.11;
    double saturationTolerance = 0.03;       // if current network performance diverts more than this tolerance from either the personal best, then stop

    
    // Check whether network is perfoming above minimum threshhold, kick out if not
    if (sa->currAccuracy < minAccuracy) {
        printf("Below minimum threshhold.");
        retrainAgent(sa);
        return;
    }
    
    validateAgent(sa);
    
    
    // Check whether network already reached target
    if (sa->validationAccuracy >= targetAccuracy) {
        printf("Target reached. CONGRATULATIONS!");
        retrainAgent(sa);
        return;
    }
    
    
    // Check whether network is generalizing
    if ( sa->currAccuracy - sa->validationAccuracy > 0)  {
        printf("Not generalizing. Validation set underperforms training set.");
        retrainAgent(sa);
        return;
    }
    
    
    // Check whether network is still making progress (i.e. not saturated)
    if ( (sa->validationAccuracy < (sa->bestAccuracy - saturationTolerance)) ) {
        printf("Not progressing. Below personal best.");
        retrainAgent(sa);
        return;
    }
    
    
    
}


void swarming(int layerCount, LayerDefinition *layerDefs){
    
    // Define swarming parameters
    int     agentCount          = 1;
    int     maxTrainTime        = 300;          // in seconds
    int     batchSize           = 5000;
    double  upperBoundary       = 0.1600;
    double  lowerBoundary       = 0;
    
    // Create the network
    Network *nn = createNetwork(layerCount, layerDefs);
    
    // Create the swarm agent and reset values
    SwarmAgent swarmAgent;
    SwarmAgent *sa = &swarmAgent;
    sa->id = 1;
    sa->nn = nn;
    sa->startRate = ((upperBoundary-lowerBoundary)/(double)2);
    sa->runCount = 0;
    
    // Open MNIST Files
    sa->imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    sa->labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    

    // Set entry point
    sa->nn->learningRate = sa->startRate;

    // Run initial batch
    outputTrainingStatsHeader();
    
    startAgent(sa);
    
    
    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);
    double runningTime = 0;

    do {
        time_t currentTime = time(NULL);
        runningTime = difftime(currentTime, startTime);
        
        evalAgent(sa);
        trainAgent(sa, sa->trainCount, sa->trainCount+batchSize);
        

    } while (runningTime<maxTrainTime);
    
    
    printf("Time is up!\n\n");

    
    fclose(sa->imageFile);
    fclose(sa->labelFile);
    
    
    
    testNetwork(sa->nn, 33);
    
    
    
}
 
 
 
 
 
 
 double validateNetwork(Network *nn, int swarmNetId, int batchSize, FILE *imageFile, FILE *labelFile){
 
 if (batchSize>10000) batchSize=10000;
 
 int errCount = 0;
 
 int validationSetStart = 50000;
 
 // Loop through all images in the file
 for (int imgId=validationSetStart; imgId<validationSetStart+batchSize; imgId++){
 
 // Reading next image and its corresponding label
 MNIST_Image img = getImageByPosition(imageFile, imgId);
 MNIST_Label lbl = getLabelByPosition(labelFile, imgId);
 
 // Convert the MNIST image to a standardized vector format and feed into the network
 Vector *inpVector = getVectorFromImage(&img);
 feedInput(nn, inpVector);
 
 // Feed forward all layers (from input to hidden to output) calculating all nodes' output
 feedForwardNetwork(nn);
 
 // Classify image by choosing output cell with highest output
 int classification = getNetworkClassification(nn);
 if (classification!=lbl) errCount++;
 
 locateCursor(20+swarmNetId, 96);
 printf("%'6d",imgId+1);
 
 }
 
 
 printf("   ");
 
 double accuracy = 1-((double)errCount/batchSize);
 
 return accuracy;
 
 }
 
 
 
 
 
 
 void trainNetwork(Network *nn, int swarmNetId, int fromImageId, int toImageId, FILE *imageFile, FILE *labelFile){
 
 if (fromImageId>=50000) return;  // fully trained already
 
 if (toImageId>50000) toImageId=50000;
 
 // Loop through all images in the file
 for (int imgId=fromImageId; imgId<toImageId; imgId++){
 
 // Reading next image and its corresponding label
 MNIST_Image img = getImageByPosition(imageFile, imgId);
 MNIST_Label lbl = getLabelByPosition(labelFile, imgId);
 
 // Convert the MNIST image to a standardized vector format and feed into the network
 Vector *inpVector = getVectorFromImage(&img);
 feedInput(nn, inpVector);
 
 // Feed forward all layers (from input to hidden to output) calculating all nodes' output
 feedForwardNetwork(nn);
 
 // Back propagate the error and adjust weights in all layers accordingly
 backPropagateNetwork(nn, lbl);
 
 locateCursor(20+swarmNetId, 63);
 printf("%'6d",imgId+1);
 
 }
 
 
 printf("   ");
 }
 
 
 

 
*/
