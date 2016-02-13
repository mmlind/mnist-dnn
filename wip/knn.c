/**
 * @file knn.c
 * @brief KNN classification
 * @author Matt Lind
 * @date January 2016
 */



/*

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>

#include "../util/mnist-utils.h"
#include "../util/screen.h"
#include "mlnn.h"
#include "knn.h"







void readTrainingImages(MNIST_Image *tImgs, MNIST_Label *tLbls){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TRAINING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        tImgs[imgCount] = getImage(imageFile);
        tLbls[imgCount] = getLabel(labelFile);
        
    }
}

void readTestingImages(MNIST_Image *tImgs, MNIST_Label *tLbls){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        tImgs[imgCount] = getImage(imageFile);
        tLbls[imgCount] = getLabel(labelFile);
        
    }
}




int vectorSubtraction2(MNIST_Image img1, MNIST_Image img2){
    
    int diff = 0;
    
    for (int p=0;p<MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT;p++){
        
        diff += pow(abs(img1.pixel[p] - img2.pixel[p]),2);
        
    }
    
    diff = sqrt(diff);
    
    return diff;
}



int vectorSubtraction(MNIST_Image img1, MNIST_Image img2){
    
    int diff = 0;
    
    for (int p=0;p<MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT;p++){
        
        diff += abs(img1.pixel[p] - img2.pixel[p]);
        
    }
    
    return diff;
}


void knnClassifier(){
    
    
    // @attention Only works for <10,000 images, otherwise memory error
    
    
    MNIST_Image trainingImages[MNIST_MAX_TRAINING_IMAGES];
    MNIST_Label trainingLabels[MNIST_MAX_TRAINING_IMAGES];
    
    MNIST_Image testingImages[MNIST_MAX_TESTING_IMAGES];
    MNIST_Label testingLabels[MNIST_MAX_TESTING_IMAGES];
    
    
    
    readTrainingImages(trainingImages, trainingLabels);
    readTestingImages(testingImages, testingLabels);
    
    int correctClassCount = 0;
    int classCount = 0;
    
    
    // Loop through all testing images and classify by calculating k-nearest training images
    
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        MNIST_Image testImg = testingImages[imgCount];
        MNIST_Label testLbl = testingLabels[imgCount];
        
        // nearest-neighbor-id default to -1
        int nnId = -1;
        int minDiff = 999999999;
        
        
        for (int t=0; t<MNIST_MAX_TRAINING_IMAGES; t++){
            
            MNIST_Image trainImg = trainingImages[t];
            
            int diff = vectorSubtraction2(testImg, trainImg);
            
            if (diff<minDiff) {
                minDiff = diff;
                nnId = t;
            }
            
        }
        
        classCount++;
        if (testLbl == trainingLabels[nnId]) correctClassCount++;
        
        double accuracy = 100*((double)correctClassCount/classCount);
        
        locateCursor(5, 5);
        printf("Tested:%3d    Correct:%3d    Accuracy:%5.2f \n\n",classCount,correctClassCount,accuracy);
        
        
        
        
    }
    
    
}

*/
