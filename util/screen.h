/**
 * @file screen.h
 * @brief Utitlies for advanced input and output to the terminal screen
 * @author Matt Lind
 * @date July 2015
 */


#ifndef SCREEN_HEADER
#define SCREEN_HEADER


// Include project libraries
#include "../dnn.h"



typedef enum Color {WHITE, RED, GREEN, YELLOW, BLUE, CYAN} Color;

static const Color DEFAULT_TEXT_COLOR = WHITE;




/**
 * @brief Clear terminal screen by printing an escape sequence
 */

void clearScreen();




/**
 * @brief Set text color in terminal by printing an escape sequence
 * @param c Color code
 */

void setColor(Color c);


/**
 * @brief Moves the cursor to the left of the current position by a specified number of steps
 * @param x Number of steps that the cursor is moved to the left
 */

void moveCursorLeft(const int x);




/**
 * @brief Moves the cursor to the specified horizontal position in the current line
 * @param x Horizaontal coordinate to why the cursor is moved to
 */

void moveCursorTo(const int x);




/**
 * @brief Set cursor position to given coordinates in the terminal window
 * @param row Row number in terminal screen
 * @param col Column number in terminal screen
 */

void locateCursor(const int row, const int col);




/**
 * @brief Outputs a summary table of the network specified via the given array of layer definitions
 * @param layerCount The number of layer in this network
 * @param layerDefs A pointer to an array of layer definitions
 */

void outputNetworkDefinition(int layerCount, LayerDefinition *layerDefs);


#endif

