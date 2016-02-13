/**
 * @file screen.c
 * @brief Utitlies for advanced input and output to the terminal screen
 * @author Matt Lind
 * @date July 2015
 */


// Include external libraries
#include <stdio.h>
#include <string.h>
#include "screen.h"

// Include project libraries
#include "../dnn.h"




/**
 * @details Clear terminal screen by printing an escape sequence
 */

void clearScreen(){
    printf("\e[1;1H\e[2J");
}




/**
 * @details Set text color in terminal by printing an escape sequence
 */

void setColor(Color c){
    char esc[5];
    strcpy(esc, "0;00");    // default WHITE
    switch (c) {
        case WHITE : strcpy(esc, "0;00");
            break;
        case RED   : strcpy(esc, "1;31");
            break;
        case GREEN : strcpy(esc, "1;32");
            break;
        case YELLOW: strcpy(esc, "1;33");
            break;
        case BLUE  : strcpy(esc, "1;34");
            break;
        case CYAN  : strcpy(esc, "1;36");
            break;
    }
    printf("%c[%sm",27,esc);
}




/**
 * @brief Moves the cursor to the specified horizontal position in the current line
 * @param x Horizaontal coordinate to why the cursor is moved to
 */

void moveCursorTo(const int x){
    printf("%cE",27);       // next line
    printf("%c[%dA",27,1);  // 1 line up
    if (x>0) printf("%c[%dC",27,x);  // move forward
}




/**
 * @brief Moves the cursor to the left of the current position by a specified number of steps
 * @param x Number of steps that the cursor is moved to the left
 */

void moveCursorLeft(const int x){
    printf("%c[%dD",27,x);
}




/**
 * @brief Set cursor position to given coordinates in the terminal window
 * @param row Row number in terminal screen
 * @param col Column number in terminal screen
 */

void locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}




/**
 * @details Returns a string for the given layer type
 */

char *getLayerTypeString(LayerType lt){
    
    if (lt==INPUT)              return "     INPUT     ";
    if (lt==CONVOLUTIONAL)      return " CONVOLUTIONAL ";
    if (lt==FULLY_CONNECTED)    return "FULLY CONNECTED";
    if (lt==OUTPUT)             return "    OUTPUT     ";
    
    return "ERROR!";
}




/**
 * @details Returns a string for the given activation type
 */

char *getActivationTypeString(ActFctType at){
    
    if (at==SIGMOID)            return "    SIGMOID    ";
    if (at==TANH)               return "     TANH      ";
    if (at==RELU)               return "     RELU      ";
    
    return "ERROR!";
}




/**
 * @brief Outputs a summary table of the network specified via the given array of layer definitions
 * @param layerCount The number of layer in this network
 * @param layerDefs A pointer to an array of layer definitions
 */

void outputNetworkDefinition(int layerCount, LayerDefinition *layerDefs){
    
    
    printf("-------------------------------");
    for (int i=0; i<layerCount;i++) printf("-----------------");
    printf("------------------\n");
    
    
    printf("Layer Index                    ");
    for (int i=0; i<layerCount;i++) printf("|       %2d       ", i);
    printf("||     TOTAL     |\n");
    
    printf("Layer Type                     ");
    for (int i=0; i<layerCount;i++) printf("| %*s", 15, getLayerTypeString((layerDefs+i)->layerType));
    printf("||               |\n");
    
    printf("-------------------------------");
    for (int i=0; i<layerCount;i++) printf("-----------------");
    printf("------------------\n");
    
    printf("Activation Function            |                ");
    for (int i=1; i<layerCount;i++) printf("| %*s", 15, getActivationTypeString((layerDefs+i)->activationType));
    printf("||               |\n");
    
    printf("Image Matrix (width x height)  ");
    for (int i=0; i<layerCount;i++) printf("|    %3d x%3d    ", (layerDefs+i)->nodeMap.width, (layerDefs+i)->nodeMap.height);
    printf("||               |\n");
    
    printf("Feature Maps (depth)           ");
    for (int i=0; i<layerCount;i++) printf("|       %2d       ", (layerDefs+i)->nodeMap.depth);
    printf("||               |\n");
    
    printf("Filter Size                    |                ");
    for (int i=1; i<layerCount-1;i++) printf("|      %d x %d     ", (layerDefs+i)->filter, (layerDefs+i)->filter);
    printf("|                ");
    printf("||               |\n");
    
    printf("Stride                         |                ");
    for (int i=1; i<layerCount-1;i++) {
        int stride = calcStride((layerDefs+i-1)->nodeMap.width, (layerDefs+i)->filter, (layerDefs+i)->nodeMap.width);
        printf("|       %2d       ", stride);
    }
    printf("|                ");
    printf("||               |\n");
    
    
    printf("-------------------------------");
    for (int i=0; i<layerCount;i++) printf("-----------------");
    printf("------------------\n");
    
    
    printf("Number of Nodes                ");
    int nodeTotal = 0;
    for (int i=0; i<layerCount;i++) {
        int nodeCount = getLayerNodeCount(layerDefs+i);
        nodeTotal += nodeCount;
        printf("|    %'9d   ", nodeCount);
    }
    printf("||   %'9d   |\n",nodeTotal);
    
    printf("Number of Connections          ");
    int connTotal = 0;
    for (int i=0; i<layerCount;i++) {
        int nodeCount = getLayerNodeCount(layerDefs+i);
        int connCount = nodeCount * getNodeBackwardConnectionCount(layerDefs+i);
        connTotal += connCount;
        printf("|   %'10d   ", connCount);
    }
    printf("||  %'10d   |\n",connTotal);
    
    
    printf("Number of Weights              ");
    int weightTotal = 0;
    for (int i=0; i<layerCount;i++) {
        int weightCount = getLayerWeightCount(layerDefs+i);
        weightTotal += weightCount;
        printf("|   %'10d   ", weightCount);
    }
    printf("||  %'10d   |\n",weightTotal);
    
    
    printf("Memory Size (bytes)            ");
    ByteSize netSize = sizeof(Network);
    for (int i=0; i<layerCount;i++) {
        ByteSize layerSize = getLayerSize(layerDefs+i);
        ByteSize weightBlock = getLayerWeightBlockSize(layerDefs+i);
        netSize += layerSize + weightBlock;
        printf("|  %'11lu   ", layerSize);
    }
    printf("|| %'11lu   |\n",netSize);
    
    
    printf("-------------------------------");
    for (int i=0; i<layerCount;i++) printf("-----------------");
    printf("------------------\n\n");
    
    
}


