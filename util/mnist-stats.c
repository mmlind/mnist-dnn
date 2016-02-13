/**
 * @file mnist-stats.c
 * @brief Utitlies for displaying details of processing the MNIST data set in the terminal screen
 * @author Matt Lind
 * @date July 2015
 */


// Include external libraries
#include <string.h>

// Include project libraries
#include "screen.h"
#include "mnist-utils.h"
#include "mnist-stats.h"




/**
 * @brief Outputs a 28x28 text frame at a defined screen position
 * @param row Row of terminal screen
 * @param col Column of terminal screen
 */

void displayImageFrame(int row, int col){
    
    if (col!=0 && row!=0) locateCursor(row, col);

    printf("------------------------------\n");
    
    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------\n");
    
}




/**
 * @brief Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 * @param img Pointer to a MNIST image
 * @param lbl Target classification
 * @param cls Actual classification
 * @param row Row on screen (x-coordinate) where to display the image
 * @param col Column on screen (y-coordinate) where to display the image
 */

void displayImage(MNIST_Image *img, int lbl, int cls, int row, int col){


    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");
    
    for (int y=0; y<MNIST_IMG_HEIGHT; y++){
        
        for (int o=0; o<col-2; o++) strcat(imgStr," ");
//        strcat(imgStr,"|");
        
        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }
    
    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
  

    printf("     Label:%d   Classification:%d\n\n",lbl,cls);
    
}




/**
 * @brief Outputs progress to the console while processing MNIST training images
 * @param imgCount Number of images already read from the MNIST file
 * @param errCount Number of errors (images incorrectly classified)
 */

void displayTrainingProgress(int imgCount, int errCount){
    
    double progress = (double)(imgCount+1)/(double)(MNIST_MAX_TRAINING_IMAGES)*100;
    
    moveCursorTo(0);
    
    printf("Training: Reading image No. %'6d of %'6d images [%3d%%]  ",(imgCount+1),MNIST_MAX_TRAINING_IMAGES,(int)progress);
    
    double accuracy = 1 - ((double)errCount/(double)(imgCount+1));
    
    printf("Result: Correct=%'6d  Incorrect=%'6d  Accuracy=%5.2f%%",imgCount+1-errCount, errCount, accuracy*100);
    
}




/**
 * @brief Outputs progress to the console while processing MNIST testing images
 * @param imgCount Number of images already read from the MNIST file
 * @param errCount Number of errors (images incorrectly classified)
 */

void displayTestingProgress(int imgCount, int errCount){
    
    double progress = (double)(imgCount+1)/(double)(MNIST_MAX_TESTING_IMAGES)*100;
    
    moveCursorTo(0);
    
    printf("Testing:  Reading image No. %'6d of %'6d images [%3d%%]  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);
    
    double accuracy = 1 - ((double)errCount/(double)(imgCount+1));
    
    printf("Result: Correct=%'6d  Incorrect=%'6d  Accuracy=%5.2f%%",imgCount+1-errCount, errCount, accuracy*100);
    
}



