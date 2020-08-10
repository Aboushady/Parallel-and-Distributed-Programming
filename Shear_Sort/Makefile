####################################################################
# Makefile for the Shear Sort Algorithm.
####################################################################

CC	= mpicc
CCFLAGS = -g
LIBS 	= -lmpi

all: ShearSort
	@echo "Usage: make Shear Sort"
ShearSort:
	$(CC) $(CCFLAGS) -o parallel_shear_sort parallel_shear_sort.c -lm $(LIBS)
clean: 
	rm parallel_shear_sort
