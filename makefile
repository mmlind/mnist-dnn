all: main

main: 
	gcc -o bin/mnist-dnn -Iutil main.c dnn.c util/screen.c util/mnist-utils.c util/mnist-stats.c -lm -std=c99

