/**
 * @file mnist-stats.h
 * @brief Utitlies for displaying details of processing the MNIST data set in the terminal screen
 * @author Matt Lind
 * @date July 2015
 */




#ifndef MNIST_STATS_HEADER
#define MNIST_STATS_HEADER



/**
 * @brief Outputs a 28x28 text frame at a defined screen position
 * @param row Row of terminal screen
 * @param col Column of terminal screen
 */

void displayImageFrame(int y, int x);




/**
 * @brief Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 * @param img Pointer to a MNIST image
 * @param lbl Target classification
 * @param cls Actual classification
 * @param row Row on screen (x-coordinate) where to display the image
 * @param col Column on screen (y-coordinate) where to display the image
 */

void displayImage(MNIST_Image *img, int lbl, int cls, int row, int col);




/**
 * @brief Outputs progress to the console while processing MNIST training images
 * @param imgCount Number of images already read from the MNIST file
 * @param errCount Number of errors (images incorrectly classified)
 */

void displayTrainingProgress(int imgCount, int errCount);




/**
 * @brief Outputs progress to the console while processing MNIST testing images
 * @param imgCount Number of images already read from the MNIST file
 * @param errCount Number of errors (images incorrectly classified)
 */

void displayTestingProgress(int imgCount, int errCount);




#endif