#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <random>
#include <algorithm>
#include <chrono>  

// Random number generator
int random(const int &min, const int &max) {
    static std::mt19937 generator(117);
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

// dot product computation with cubic root
double dot_product(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return std::cbrt(result); // Taking cubic root of the dot product
}

// Sequential wavefront computation
void sequential_wavefront(std::vector<double> &M, const uint64_t &N) {
    for (uint64_t k = 0; k < N; ++k) {  // For each upper diagonal
        for (uint64_t i = 0; i < (N - k); ++i) {  //For each element in the diagonal
            std::vector<double> v1(k), v2(k);
            for (uint64_t j = 0; j < k; ++j) {
                v1[j] = M[i * N + j];
                v2[j] = M[(i + k) * N + j];
            }
            M[i * N + (i + k)] = dot_product(v1, v2);
        }
    }
}

// comparing Sequential and parallel matrix
bool compare_matrices(const std::vector<double> &M1, const std::vector<double> &M2) {

    return M1==M2; // all elements are equal
}

// parallel wavefront computation
void parallel_wavefront_computation(std::vector<double> &M, const uint64_t &N, int rank, int size) {
    uint64_t rows_per_proc = N / size;  //number of rows each process will handle
    uint64_t remainder = N % size;

    uint64_t start_row = rank * rows_per_proc + std::min(static_cast<uint64_t>(rank), remainder);
    uint64_t end_row = start_row + rows_per_proc;
    if (static_cast<uint64_t>(rank) < remainder) end_row++;

    for (uint64_t k = 0; k < N; ++k) {  // each upper diagonal
        MPI_Barrier(MPI_COMM_WORLD);  // synchronize all processes before computing each diagonal

        for (uint64_t i = start_row; i < end_row && i < (N - k); ++i) {  //each element in the diagonal
            std::vector<double> v1(k), v2(k);
            for (uint64_t j = 0; j < k; ++j) {
                v1[j] = M[i * N + j];
                v2[j] = M[(i + k) * N + j];
            }
            M[i * N + (i + k)] = dot_product(v1, v2);
            // print the result computed by this node
//            std::cout << "Rank " << rank << " computed M[" << i << "][" << (i + k) << "] = " << M[i * N + (i + k)] << "\n";
        }

        // send data to the next process 
        if (rank < size - 1 && end_row < (N - k)) {
            MPI_Send(&M[(end_row - 1) * N + k], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }

        // receive data from the previous process 
        if (rank > 0 && start_row < (N - k)) {
            MPI_Recv(&M[(start_row - 1) * N + k], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);  // synchronize all processes after computing each diagonal
    }
}


// print matrix values with indices 
void print_matrix_index(const std::vector<double> &M, const uint64_t &N, const std::string &name, int rank) {
    std::cout << "Matrix " << name << " from rank " << rank << ":\n";
    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            std::cout << "M[" << i << "][" << j << "] = " << M[i * N + j] << "\n";
        }
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int min = 0;        // Default minimum random  value
    int max = 1000;     // Default maximum  random  value
    uint64_t N = 10;   // Default size of the matrix (NxN)

    // Parse command line arguments
    if (argc != 1 && argc != 2 && argc != 4) {
        if (rank == 0) {
            std::printf("use: %s N [min max]\n", argv[0]);
            std::printf("     N size of the square matrix\n");
            std::printf("     min  \n");
            std::printf("     max  \n");
        }
        MPI_Finalize();
        return -1;
    }
    if (argc > 1) {
        N = std::stol(argv[1]);
        if (argc > 2) {
            min = std::stol(argv[2]);
            max = std::stol(argv[3]);
        }
    }

    // Allocate and initialize the matrix
    std::vector<double> M_seq(N * N, -1.0); // For sequential computation
    std::vector<double> M_parl(N * N, -1.0); // For parallel computation

    // Initialize the main diagonal and other elements
    for (uint64_t i = 0; i < N; ++i) {
        M_seq[i * N + i] = static_cast<double>(i + 1) / N;
        M_parl[i * N + i] = static_cast<double>(i + 1) / N;
    }

    // Function to initialize the matrix with random values
    auto init = [&](std::vector<double> &M) {
        for (uint64_t k = 1; k < N; ++k) {
            for (uint64_t i = 0; i < (N - k); ++i) {
                int t = random(min, max);
                M[i * N + (i + k)] = static_cast<double>(t);
            }
        }
    };

    init(M_seq); // Initialize the sequential matrix
    init(M_parl); // Initialize the parallel matrix

    if (rank == 0) {
        std::printf("Matrix initialized.\n");
    }

    // Starting timing for sequential computation 
    auto start_seq = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        sequential_wavefront(M_seq, N);
        // end timing for parallel computation
        auto end_seq = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> seq_duration = end_seq - start_seq;
        std::printf("Sequential computation took %f seconds.\n", seq_duration.count());
        // Print sequential matrix result
//        print_matrix_index(M_seq, N, "Sequential", rank);
    }

    // Broadcast the initialized matrix to all processes
    MPI_Bcast(M_parl.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Starting timing for parallel computation
    MPI_Barrier(MPI_COMM_WORLD); // Ensureing all processes start together
    auto start_parl = std::chrono::high_resolution_clock::now();
    
    parallel_wavefront_computation(M_parl, N, rank, size);

    // end timing for parallel computation
    auto end_parl = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mpi_duration = end_parl - start_parl;
    double mpi_time = mpi_duration.count();  
    std::printf("Rank %d took %f seconds for parallel computation.\n", rank, mpi_time);

    // Calculate the total parallelization time by taking the max time from all ranks
    double total_parallel_time;
    MPI_Reduce(&mpi_time, &total_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // gather the results from all processes to process 0
    uint64_t rows_per_proc = N / size;
    uint64_t remainder = N % size;
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    uint64_t start_row = rank * rows_per_proc + std::min(static_cast<uint64_t>(rank), remainder);

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (rows_per_proc + (static_cast<uint64_t>(i) < remainder ? 1 : 0)) * N;
        displs[i] = (i > 0) ? displs[i - 1] + recvcounts[i - 1] : 0;
    }

    MPI_Gatherv(M_parl.data() + start_row * N, recvcounts[rank], MPI_DOUBLE, 
                M_parl.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // print the final parallel matrix from each process
  //  print_matrix_index(M_parl, N, "Parallel", rank);

    // compare results of sequential and parallel computations
    if (rank == 0) {
        if (compare_matrices(M_seq, M_parl)) {
            std::printf("The results of the sequential and MPI computations match.\n");
        } else {
            std::printf("The results of the sequential and MPI computations do not match.\n");
        }
        std::printf("Total parallel computation took %f seconds.\n", total_parallel_time);
    }

    MPI_Finalize();
    return 0;
}
