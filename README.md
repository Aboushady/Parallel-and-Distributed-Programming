# Parallel-and-Distributed-Programming
Implementing the following algorithms in C and OpenMPI:

1- Parallel Matrix-Matrix Multiplication.
2- Parallel Shear Sort.


# Requirments (What I am using):
1- OpenMPI.
2- gcc.
3- Linux, Ubuntu 18.4.

# Parallel Matrix-Matrix Multiplication:
## Build:
1- Open shell/command line.
2- Navigate to the diroctory, where this project is saved.
3- type "make", this will run the set of directives saved in the "Makefile" file, to compile and link the source code.
## Run:
To run this parallel program with 3 cores, and an input file that contain two matrices of order 3 to be multiplied:
1- Type "mpirun -3 matmul input3.txt output.txt".
2- The result matrix will be saved in "the output.txt".


# Parallel Shear Sort:
## Build:
1- Open shell/command line.
2- Navigate to the diroctory, where this project is saved.
3- type "make", this will run the set of directives saved in the "Makefile" file, to compile and link the source code.
## Run:
To run this paralll program with 4 cores and an input file that contains an unsorted matrix of order 4:
1- While in the same directory, type "mpirun -4 parallel_shear_sort unsorted.txt".
2- This will generate a text fie called "output4_4.txt", that contains the matrix sorted in a snake-like order.
