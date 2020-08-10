####################################################################
# Makefile for the Parallel Matrix Multiplication Algorithm
#
####################################################################

CC	= mpicc
CCFLAGS = -g
LIBS 	= -lmpi

all: quicksort
	@echo "Usage: make Matrix Multiplication"
quicksort:
	$(CC) $(CCFLAGS) -o matmul matmul.c $(LIBS)
clean: 
	rm matmul
