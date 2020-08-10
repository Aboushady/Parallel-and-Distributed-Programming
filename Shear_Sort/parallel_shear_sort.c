#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <memory.h>
#include <time.h>

// Decalre functions.
int comfunc(const void *p, const void *q);
void odd_even_row_sort(int local_n, int my_rank, double local_a[local_n], int comm_size, int row_per_p);
void merge_low(int local_n, double local_a[local_n], double new_local_a[local_n], int row_per_p);
void merge_high(int local_n, double local_a[local_n], double new_local_a[local_n], int row_per_p, int my_rank);

void main(int argc, char *argv[]){
    // Declare rank, communicatation pool size, and size of the local matrix.
    int my_rank, comm_size, local_n, n, row_per_p;

    // Time variables:
    clock_t start_t, end_t; 
    double total_t;
    struct timespec ts1, ts2;

    // "Key", will hold the value of matrix's elements read form the file.
    double key;

    // Initialize MPI, rank, and size.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (my_rank == 0){
        FILE *fp;
        char *fname;
        fname = argv[1];

        //Open the file.        
        fp = fopen(fname, "r"); // read mode.

        // Check if the file were opened successfully.
        if (fp == NULL){
            perror("Error while opening the file.\n");
            exit(EXIT_FAILURE);
        }        
        fscanf(fp, "%d", &n);
        fclose(fp);
    }

    // Force all processes to wait till every process arrives here.
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast "n" to all processes.
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    // Calculate local_n.
    local_n = (n*n)/comm_size;
    row_per_p = local_n/n;
    // Initialize local_a and declare *a .
    double local_a[local_n];
    double *a = NULL;

    if (my_rank == 0){ 
        FILE *fp;
        char *fname;
        fname = argv[1];
        fp = fopen(fname, "r");
        fscanf(fp, "%d", &n);
        
        //Dynamically allocate a[n][n].
        a = (double *) malloc((n*n) * sizeof(double));

        for (int i=0; i<n*n; i++){
            fscanf(fp, "%lf", &key);
            a[i] = key;                    
        }
        fclose(fp);
    }

    // Force all processes to wait till every process arrives here.
    MPI_Barrier(MPI_COMM_WORLD);


    // Initialize start time.
    start_t = clock();
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    // Send a number of local_n of elements to each process.
    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // initialize "d" number of maximum phases.
    int d;
    int d_log_2 = log10(n) / log10(2); // Convert log_10 to log_2. 
    d = ceil(d_log_2);

    // sort local_a of each process, in ascending order.
    qsort((void*)local_a, local_n, sizeof(double), comfunc);


    // Initialize a_transposed.
    double *a_transposed = NULL;

    // Iterate through the phases.
    for(int phase=0; phase < d; phase++){
        if(phase % 2 == 0){
            odd_even_row_sort(local_n, my_rank, local_a, comm_size, row_per_p);

            // Force all processes to wait till every process arrives here.
            MPI_Barrier(MPI_COMM_WORLD);
            
            // a <- gather elements back to root.
            MPI_Gather(local_a, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        } else {
            // Transpose elements of "a", and assign to "a_transposed".
            // "a_transposed" -> holds the transposed matrix.            
            if (my_rank == 0){
                a_transposed = (double *) malloc((n*n) * sizeof(double));

                int i, j;
                int counter = 0;
                for (i=0; i<n; i++){
                    for (j = 0; j < n; j++){
                        a_transposed[j+counter*i] = a[j*n+i];
                    }
                    counter = n; 
                }    
            }

            // Force all processes to wait till every process arrives here.
            MPI_Barrier(MPI_COMM_WORLD);

            // Scatter each row of a_transposed "holds transposed elements" to processes.
            MPI_Scatter(a_transposed, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            free(a_transposed);

            // Sort each local_a ascendingly.
            qsort((void*)local_a, local_n, sizeof(double), comfunc);

            // Transposed elements are gathered back and saved in "a".
            MPI_Gather(local_a, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Force all processes to wait till every process arrives here.
            MPI_Barrier(MPI_COMM_WORLD);

            // Transpose back to original and assign to "a_transposed".
            // "a_transposed" -> will now hold the elements in original form.
            if (my_rank == 0) {
                a_transposed = (double *) malloc((n*n) * sizeof(double));
                
                int i, j;
                int counter = 0;
                for (i=0; i<n; i++){
                    for (j = 0; j < n; j++){
                        a_transposed[j+counter*i] = a[j*n+i];
                    }
                    counter = n;
                }                
                qsort((void*)a_transposed, n*n, sizeof(double), comfunc);
            }

            // Force all processes to wait till every process arrives here.
            MPI_Barrier(MPI_COMM_WORLD);

            // Scatter each row of a_transposed "holds elements in original form" to processes.
            MPI_Scatter(a_transposed, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Sort each local_a ascendingly.
            //qsort((void*)local_a, local_n, sizeof(double), comfunc);

            free(a_transposed);
        }

    }
    if (d % 2 == 0){
        odd_even_row_sort(local_n, my_rank, local_a, comm_size, row_per_p);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(local_a, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize end_time.
    end_t = clock();
    clock_gettime(CLOCK_MONOTONIC, &ts2);

    if (my_rank == 0){
        char fname[100];
        FILE *fp;

        sprintf(fname,"output%d_%d.txt", local_n, comm_size);
        fp = fopen(fname, "w");
        
        // For formatting the ouput.
        int row_break = 1;
        
        for (int i = 0; i < n * 3; i++){ 
            
            if (i  == 0){
                fprintf(fp, "C :\n");
            }

            fprintf(fp, "%lf ", a[i]);
            
            if (row_break == n){
                fprintf(fp, "\n");
                row_break = 0;
            }
            row_break++;      
        }

                // Write exexution time in the file. FOR NUMERICAL EXPERMINTS. CHANGE IT AFTER.
        double time_taken;
        time_taken = ((double)end_t -start_t) / CLOCKS_PER_SEC;
         
        total_t = (ts2.tv_sec - ts1.tv_sec) * 1e9; 
        total_t = (total_t + (ts2.tv_nsec - ts1.tv_nsec)) * 1e-9; 
        fprintf(fp, "Real execution time in seconds : %f\n", total_t);
        fprintf(fp, "Clock execution time in seconds : %f\n", time_taken);
        fclose(fp);
    }
    
    // Wait for all process, to exit together.
    MPI_Barrier(MPI_COMM_WORLD);

    // Freeing pointers.
    free(a);

    // Close MPI.
    MPI_Finalize();

} // Main [END]

// Compares two elements, and sort them ascendingly.
int comfunc(const void *p, const void *q){
    //return (*(const double*)p - *(const double*)q);
    if (*(double*)p > *(double*)q)
        return 1;
    else if (*(double*)p < *(double*)q)
        return -1;
    else
        return 0;  
} // Compare Function [END]


/* To force a bahaviour of even ranked processes sending to odd ranked processes only.
    P0 -> P1, P2 -> P3, if n = 4.
*/
void odd_even_row_sort(int local_n, int my_rank, double local_a[local_n], int comm_size, int row_per_p){
    double new_local_a[local_n];
    int n = local_n/row_per_p;

    // If it's the last process and comm_size is odd, don't communicate with any process,
    // And just order it's elements.
    if (my_rank+1 < comm_size || comm_size % 2 == 0){
        if(my_rank % 2 == 0){
            MPI_Send(local_a, local_n, MPI_DOUBLE, (my_rank+1) % comm_size, 0, MPI_COMM_WORLD);
            MPI_Recv(new_local_a, local_n, MPI_DOUBLE, (my_rank+1) % comm_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            merge_low(local_n, local_a, new_local_a, row_per_p);
        } else {
            MPI_Recv(new_local_a, local_n, MPI_DOUBLE, (my_rank+comm_size-1) % comm_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);        
            MPI_Send(local_a, local_n, MPI_DOUBLE, (my_rank+comm_size-1) % comm_size, 0, MPI_COMM_WORLD);
            merge_high(local_n, local_a, new_local_a, row_per_p, my_rank);
        }
    } else {
        /*
            [BEGIN]
            THIS PART IS CALLED "SHEAR".
            E.g. : local_a = [0, 1, 2, 3, 4, 5, 6, 7, 8], n = 3, i = 0, ..., n-1.
            if (i = 0):
                temp[0] = local_a[0], temp[1] = local_a[1], temp[2] = local_a[3] => Ascending order.
            if (i = 1):
                 temp[3] = local_a[5], temp[4] = local_a[4], temp[5] = local_a[3] => Descending order.
            and so on...
        */

        double *temp = (double *) malloc(local_n * sizeof(double));
        for(int i=0; i<row_per_p; i++){
            for (int j = 0; j <n; j++){
                if (i % 2 == 0){
                    temp[i*n+j] = local_a[i*n+j];
                } else {
                    temp[i*n+j] = local_a[((i+1)*n)-1-j];
                }
            }
        }

        for(int i = 0; i<local_n; i++){
            local_a[i] = temp[i];
        }

        /*END*/

        free(temp);
    }

} /*Odd_even_row_sort [END]*/


/*
    This is method is called by the even procosses rank(P) % 2 == 0, 
    to compare their local_n's elements to local_n's elements received from rank(P+1) % 2 != 0.
    And keep the smallest n elements.

    This method depends on local_a's and new_local_a's, being ordered ascendingly.
    So OUTPUT the new local_a will be ordered ascendingly. 
*/
void merge_low(int local_n, double local_a[local_n], double new_local_a[local_n], int row_per_p){
    double temp_local_a[local_n];
    double *temp = (double *) malloc(local_n * sizeof(double));
    int m_i, r_i, t_i, n;
    m_i = 0, r_i = 0, t_i = 0, n = local_n/row_per_p;


    while(t_i < local_n){
        if (local_a[m_i] <= new_local_a[r_i]){
            temp_local_a[t_i] = local_a[m_i];
            t_i++; m_i++;

        } else{
            temp_local_a[t_i] = new_local_a[r_i];
            t_i++; r_i++;
        }
    }

    /*
        [BEGIN]
        Another usage of the "SHEAR" part. In case P < N, that is each process has more than one row.
    */
    for(int i=0; i<row_per_p; i++){
        for (int j = 0; j <n; j++){
            if (i % 2 == 0){
                temp[i*n+j] = temp_local_a[i*n+j];
            } else {
                temp[i*n+j] = temp_local_a[((i+1)*n)-1-j]; //[i*n+2-j];
            }
        }
    }   /*[END]*/

    // Assign the smallest n elements to local_n.
    for (m_i = 0; m_i < local_n; m_i++){
        local_a[m_i] = temp[m_i];
    }

    free(temp);

} /*Merge low [END]*/


/*
    This is method is called by the ODD procosses rank(P) % 2 != 0, 
    to compare their local_n's elements to local_n's elements received from rank(P-1) % 2 == 0.
    And keep the largest n elements.

    This method depends on local_a's and new_local_a's, being ordered ascendingly.
    So OUTPUT the new local_a will be ordered descendingly.  
*/
void merge_high(int local_n, double local_a[local_n], double new_local_a[local_n], int row_per_p, int my_rank){
    double temp_local_a[local_n];
    double *temp = (double *) malloc(local_n * sizeof(double));
    int m_i, r_i, t_i, n;
    m_i = local_n-1, r_i = local_n-1, n = local_n/row_per_p;
    t_i = local_n-1;

    while(t_i >= 0){
        if (local_a[m_i] >= new_local_a[r_i]){
            temp_local_a[t_i] = local_a[m_i];
            t_i--; m_i--;

        } else{
            temp_local_a[t_i] = new_local_a[r_i];
            t_i--; r_i--;
        }
    }

    /*
        [BEGIN]
        Another usage of the "SHEAR" part, with an extension as follows:
        if n = 6, comm_s = 3, then each p will have 12 elements, and p1 ordering should be,
        ASC order for 6 elements, then DES order for the other 6.
        However, if n = 9, comm_s = 3, then each p will have 21 elements, and p1 ordering should be,
        DSC ordering for 9 elements, then ASC for the next 9, then DSC again for the last 9. 
    */    
    if ((my_rank*row_per_p) % 2 == 0){
        // ASC order
        for(int i=0; i<row_per_p; i++){            
            for (int j = 0; j <n; j++){                
                if (i % 2 == 0){
                    temp[i*n+j] = temp_local_a[i*n+j];
                } else {
                    temp[i*n+j] = temp_local_a[((i+1)*n)-1-j];
                }
            }
        } 
    } else {
        // DSC ordr
        for(int i=0; i<row_per_p; i++){            
            for (int j = 0; j <n; j++){                
                if (i % 2 == 0){                    
                    temp[i*n+j] = temp_local_a[((i+1)*n)-1-j];
                    // temp[i*n+j] = temp_local_a[i*n+2-j];
                } else {
                    temp[i*n+j] = temp_local_a[i*n+j];
                }
            }
        }
    }
    /*[END]*/



    // Assign the new elements to local_a.
    for (m_i = 0; m_i < local_n; m_i++){
        local_a[m_i] = temp[m_i];
    }    

    free(temp);
} /*Merge _high [END]*/
