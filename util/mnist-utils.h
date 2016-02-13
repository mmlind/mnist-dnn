/**
 * @file mnist-utils.h
 * @brief Utitlies for handling the MNIST data set files
 * @see http://yann.lecun.com/exdb/mnist/
 * @author Matt Lind
 * @date July 2015
 */

#ifndef MNIST_UTILS_HEADER
#define MNIST_UTILS_HEADER




// Include external libraries
#include <stdint.h>
#include <stdio.h>

// Define locations of MNIST data set files
#define MNIST_TRAINING_SET_IMAGE_FILE_NAME "./data/train-images-idx3-ubyte"
#define MNIST_TRAINING_SET_LABEL_FILE_NAME "./data/train-labels-idx1-ubyte"
#define MNIST_TESTING_SET_IMAGE_FILE_NAME "./data/t10k-images-idx3-ubyte"
#define MNIST_TESTING_SET_LABEL_FILE_NAME "./data/t10k-labels-idx1-ubyte"

/// Define number datasets (images+labels) in the TRAIN file/s
#define MNIST_MAX_TRAINING_IMAGES 60000

/// Define number datasets (images+labels) in the TEST file/s
#define MNIST_MAX_TESTING_IMAGES 10000

// Define image size in pixels
#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28



typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;


typedef struct Vector Vector;




/**
 * @brief Variably-sized data structure defining a vector with "count" doubles
 */

struct Vector{
    int count;              // number of values in the vector
    double vals[];          // array of values inside the vector
};




/**
 * @brief Data block defining a MNIST image
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */
struct MNIST_Image{
    uint8_t pixel[MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT];
};




/**
 * @brief Data block defining a MNIST image file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first image.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};




/**
 * @brief Data block defining a MNIST label file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first label.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};




/**
 * @brief Returns a file pointer to the MNIST image file
 * @details Opens the file and moves the read pointer to the position of the 1st image
 * @see http://yann.lecun.com/exdb/mnist/ for more details on the file definition
 */

FILE *openMNISTImageFile(char *fileName);




/**
 * @brief Returns a file pointer to the MNIST label file
 * @details Opens the file and moves the read pointer to the position of the 1st label
 * @see http://yann.lecun.com/exdb/mnist/ for more details on the file definition
 */

FILE *openMNISTLabelFile(char *fileName);



/**
 * @brief Returns the next image in given MNIST image file
 */

MNIST_Image getImage(FILE *imageFile);




/**
 * @details Returns the image located at the specified position in the given MNIST image file
 */

MNIST_Image getImageByPosition(FILE *imageFile, int position);



/**
 * @brief Returns the next label in given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile);




/**
 * @details Returns the label located at the specified position in the given MNIST label file
 */

MNIST_Label getLabelByPosition(FILE *labelFile, int position);




/**
 * @brief Returns a Vector holding the image pixels of a given MNIST image
 * @param img A pointer to a MNIST image
 */

Vector *getVectorFromImage(MNIST_Image *img);




#endif 



