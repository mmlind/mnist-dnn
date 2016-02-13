/**
 * @file mnist-utils.c
 * @brief Utitlies for handling the MNIST data set files
 * @see http://yann.lecun.com/exdb/mnist/
 * @author Matt Lind
 * @date July 2015
 */

// Include external libraries
#include <stdlib.h>
#include <string.h>

// Include project libraries
#include "mnist-utils.h"



/**
 * @details Reverse byte order in 32bit numbers
 * MNIST files contain all numbers in reversed byte order,
 * and hence must be reversed before using
 */

uint32_t flipBytes(uint32_t n){
    
    uint32_t b0,b1,b2,b3;
    
    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;
    
    return (b0 | b1 | b2 | b3);
    
}




/**
 * @details Read MNIST image file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){
    
    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;
    
    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);
    
    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);
    
    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);
    
    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}




/**
 * @details Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){
    
    lfh->magicNumber =0;
    lfh->maxImages   =0;
    
    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);
    
    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);
    
}




/**
 * @brief Returns a file pointer to the MNIST image file
 * @details Opens the file and moves the read pointer to the position of the 1st image
 * @see http://yann.lecun.com/exdb/mnist/ for more details on the file definition
 */

FILE *openMNISTImageFile(char *fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);
    
    return imageFile;
}




/**
 * @brief Returns a file pointer to the MNIST label file
 * @details Opens the file and moves the read pointer to the position of the 1st label
 * @see http://yann.lecun.com/exdb/mnist/ for more details on the file definition
 */

FILE *openMNISTLabelFile(char *fileName){
    
    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);
    
    return labelFile;
}




/**
 * @details Returns the next image in the given MNIST image file
 */

MNIST_Image getImage(FILE *imageFile){
    
    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }
    
    return img;
}




/**
 * @details Returns the image located at the specified position in the given MNIST image file
 */

MNIST_Image getImageByPosition(FILE *imageFile, int position){
    
    MNIST_Image img;
    size_t result;
    fseek(imageFile, sizeof(MNIST_ImageFileHeader) + (position * sizeof(img)), SEEK_SET);
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }
    
    return img;
}




/**
 * @details Returns the next label in the given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile){
    
    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }
    
    return lbl;
}




/**
 * @details Returns the label located at the specified position in the given MNIST label file
 */

MNIST_Label getLabelByPosition(FILE *labelFile, int position){
    
    MNIST_Label lbl;
    size_t result;
    fseek(labelFile, sizeof(MNIST_LabelFileHeader) + (position * sizeof(lbl)), SEEK_SET);
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }
    
    return lbl;
}




/**
 * @brief Returns a Vector holding the image pixels of a given MNIST image
 * @param img A pointer to a MNIST image
 */

Vector *getVectorFromImage(MNIST_Image *img){
    
    Vector *v = (Vector*)malloc(sizeof(Vector) + (MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT * sizeof(double)));
    
    v->count = MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT;
    
    for (int i=0;i<v->count;i++)
        //        v->vals[i] = ((double)img->pixel[i]/255);     // Image pixels' grey shades (0-255) are normalized to 0-1
        // Pre-processing the input data: subtract mean and normalize
        v->vals[i] = ((double)(img->pixel[i]-127)/128);
    
    return v;
}

