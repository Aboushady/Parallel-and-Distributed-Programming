#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <memory.h>
#include <time.h>


double mat_mult(int local_n, int row_per_p, double local_a[local_n], double local_b[local_n]);

void main (int argc, char *argv[]){
    // For time Measurements.
    struct timespec ts1, ts2;
    

    double key;
    int my_rank, comm_size, local_n, row_per_p;
    int n[1];
    clock_t start_t, end_t; 
    double total_t;
    
    // Initialze MPI variables.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // read the value of n.
    if (my_rank == 0){
        FILE *fp;
        char *fname;
        fname = argv[1];
        fp = fopen(fname, "r");
        fscanf(fp, "%d", n);
        
        fclose(fp);
    }
    
    // For all processes to wait for process 0.    
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast n to all processes.
    MPI_Bcast(n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    local_n = (n[0]*n[0])/comm_size;
    row_per_p = local_n/n[0];
    
    // New 7/6/2020 [begin]
    double local_a[local_n], local_b[n[0]];
    // New [end]


    double b[n[0] * n[0]];//, a[local_n * local_n];
    double *a = NULL;
    int b_row_counter;


    // Read a and b's elements from the text file.
    if (my_rank == 0){
        // New 6/6/2020
        double *temp_b = NULL;
        temp_b = (double *)malloc((n[0]*n[0]) * sizeof(double));
        a = (double *)malloc((n[0]*n[0]) * sizeof(double));

        int i, j;

        // b_row_counter is to keep track of which row we are scattering to all processes.
        b_row_counter = 0;

        FILE *fp;
        char *fname;
        fname = argv[1];
        fp = fopen(fname, "r");
        fscanf(fp, "%d", n);

        // The counter is used to seperate what values are being read, a's values, then b's values.
        int counter = 2;
        while (counter > 0){
            for (i=0; i<n[0]*n[0]; i++){                
                fscanf(fp, "%lf", &key);
                if (counter == 2){
                    a[i] = key;
                } else {
                    temp_b[i] = key;                                
                }
            }
           counter--;
        }
        
        // transpose b from 1-dim array that represents a 2-dim array.
        // using this equation (row*number_of_columns+column).
        counter = 0;
        for (i=0; i<n[0]; i++){
            for (j = 0; j < n[0]; j++){
                b[j+counter*i] = temp_b[j*n[0]+i];
            }
            counter = n[0]; // to get the 2nd and 3rd indices in b.
        }

        free(temp_b);
        fclose(fp);

        /*
        // Get only  the b_row_counter's row of transposed b, and assign it to local_b.
        // By following these steps:
        // b_row_counter iterate over local_n, e.g. from 0,..., local_n-1.
        // we consider the array to be divided into local_n parts.
        // but since a 2-dim array is saved in the form of 1-dim array.
        // We divide this 1-dim array like this.
        // i = 0, .... , local_n-1.
        // to get the first part where b_row_counter = 0.
        // our iterator will look like this, b[i+(b_row_counter*local_n)].
        // So if we have b = [1, 2, 3, 4], local_n = 2, b_row_counter get a value of [0, 1], i = 0, 1.
        // we wanna split b into local_n parts, e.g. local_b = [1, 2], then local_b = [3, 4].
        // we get [1, 2] by e.g. i = 0, b_row_counter = 0, then local_b[i]=  b[i+(b_row_counter*local_n)],
        // = b[0+(0*2)] = b[0] , then i = 1, b_row_counter = 0, we get local_b[1] = b[1+(0*2)] = b[1].
        // the second part of b we get, e.g. i = 0, b_row_counter = 1, then local_b[i] = b[i+(b_row_counter*local_n)],
        // = b[0+(1*2)] = b[2], and when i = 1, we get local_b[1] = b[1+(1*2)] = b[3].
        // so first division local_b = [1, 2], and in the second local_b = [3, 4]. 
        */

        for (i = 0; i < n[0]; i++){
            local_b[i] = b[i+(b_row_counter*n[0])];
        } // Now, only process 0 has b_row.        
    }



    // Block all processes, untill all of them reaches here.
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Initialize start time.
    start_t = clock();
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    // Scatter each row of a to each process.
    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast local_b to all processes.
    MPI_Bcast(local_b, n[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Here we treat the matrix matrix mult, as a matrix vetor multi, and do the multiplication,
    // between a and each column of b "vector".
    int local_i;
    double local_c[local_n];
    local_i = 0;
    int counter = 0;

    for (int i = 0; i < local_n/row_per_p; i++){
        for (int j=0; j<row_per_p; j++){
            local_c[j+counter*i] = mat_mult(local_n/row_per_p, j, local_a, local_b); 
        }
        counter = row_per_p;
        if (local_i < n[0]-1){
            if (my_rank == 0 ){
                // Get only the b_row_counter's row of b, and assign it to b_row.
                for (int j = 0; j < n[0]; j++){
                    if (j == 0){
                        b_row_counter = (b_row_counter+1) % n[0];    
                    }
                    // New 6/6/2020 [begin]
                    local_b[j] = b[j+(b_row_counter*n[0])];
                    // New [end]
                } // Now, only process 0 has local_b.
                

            }
            // Block all processes, untill all of them reaches here.
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Bcast(local_b, n[0], MPI_DOUBLE, 0, MPI_COMM_WORLD); // Now, all other processes has local_b. 
        }
        local_i++;
    }


    

    // Block all processes, untill all of them reaches here.
    MPI_Barrier(MPI_COMM_WORLD);

    
    // Print the values of C to check for correctness.
    if (my_rank == 0){
        // Initialize end_time.
        end_t = clock();
        clock_gettime(CLOCK_MONOTONIC, &ts2);


        // Get Execution time.
        double time_taken;
        time_taken = ((double)end_t -start_t) / CLOCKS_PER_SEC;
         
        total_t = (ts2.tv_sec - ts1.tv_sec) * 1e9; 
        total_t = (total_t + (ts2.tv_nsec - ts1.tv_nsec)) * 1e-9; 
        
        // Only the time in seconds will be written to stdout.
        fprintf(stdout, "%f\n", total_t);

    } 

    // Block all processes, untill all of them reaches here.
    MPI_Barrier(MPI_COMM_WORLD);

    
    // Block all processes, untill all of them reaches here.
    MPI_Barrier(MPI_COMM_WORLD);

    // End MPI.
    MPI_Finalize();
}

double mat_mult(int local_n, int row_per_p, double local_a[local_n], double local_b[local_n]){
    double temp = 0;
    for (int i =0; i < local_n; i++){
        temp = temp + (local_a[i+local_n*row_per_p]*local_b[i]);
    }
    return temp;
}