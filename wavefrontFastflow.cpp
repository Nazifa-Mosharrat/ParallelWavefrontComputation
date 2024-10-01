#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>  
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

int random(const int &min, const int &max) {
    static std::mt19937 generator(117);
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

// dot product computation
double dot_product(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    double result = 0.0;
    for(size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return std::cbrt(result);   // Taking cubic root of the dot product
}

// Sequential wavefront computation
void sequential_wavefront(std::vector<double> &M, const uint64_t &N) {
    for(uint64_t k = 0; k < N; ++k) { // for each upper diagonal
        for(uint64_t i = 0; i < (N - k); ++i) { // for each element in the diagonal
            std::vector<double> v1(k), v2(k);
            for(uint64_t j = 0; j < k; ++j) {
                v1[j] = M[i * N + j];
                v2[j] = M[(i + k) * N + j];
            }
            M[i * N + (i + k)] = dot_product(v1, v2);
        }
    }
}
// Function to compare the results of the sequential and parallel computations
bool compare_results(const std::vector<double> &M_seq, const std::vector<double> &M_parl) {
    return M_seq == M_parl;
}

// Parallel wavefront computation
void parallel_wavefront(std::vector<double> &M, const uint64_t &N) {
    ParallelFor pf;
    for(uint64_t k = 0; k < N; ++k) { // for each upper diagonal
        pf.parallel_for(0, N - k, 1, 0, [&M, N, k](const long i) {
            std::vector<double> v1(k), v2(k);
            for(uint64_t j = 0; j < k; ++j) {
                v1[j] = M[i * N + j];
                v2[j] = M[(i + k) * N + j];
            }
            M[i * N + (i + k)] = dot_product(v1, v2);
        });
    }
}


int main(int argc, char *argv[]) {
    int min = 0;         // Default minimum random  value
    int max = 1000;      // Default maximum  random  value
    uint64_t N = 512;    // Default size of the matrix (NxN)

    if (argc != 1 && argc != 2 && argc != 4) {
        std::printf("use: %s N [min max]\n", argv[0]);
        std::printf("     N size of the square matrix\n");
        std::printf("     min  \n");
        std::printf("     max \n");
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
    std::vector<double> M_seq(N * N, -1.0);
    std::vector<double> M_parl(N * N, -1.0);

    // Initialize the main diagonal and other elements
    for(uint64_t i = 0; i < N; ++i) {
        M_seq[i * N + i] = static_cast<double>(i + 1) / N;
        M_parl[i * N + i] = static_cast<double>(i + 1) / N;
    }

    auto init = [&](std::vector<double> &M) {
        for(uint64_t k = 1; k < N; ++k) {
            for(uint64_t i = 0; i < (N - k); ++i) {
                int t = random(min, max);
                M[i * N + (i + k)] = static_cast<double>(t);
            }
        }
    };

    init(M_seq);
    init(M_parl);

    std::printf("Matrix initialized.\n");

    //sequential wavefront computation

    // Starting timing for sequential computation 
     auto start_seq = std::chrono::high_resolution_clock::now();   
  
    
    sequential_wavefront(M_seq, N);
     // end timing for parallel computation
     auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seq_duration = end_seq - start_seq;
    std::printf("Sequential computation took %f seconds.\n", seq_duration.count());

    // parallel wavefront computation
    // Starting timing for parallel computation
    auto start_parl = std::chrono::high_resolution_clock::now();
    parallel_wavefront(M_parl, N);
    // end timing for parallel computation
    auto end_parl = std::chrono::high_resolution_clock::now();   
    std::chrono::duration<double> ff_duration = end_parl - start_parl;
    std::printf("parallel computation took %f seconds.\n", ff_duration.count());
 
     // compare results of sequential and parallel computations
    if (compare_results(M_seq, M_parl)) {
        std::printf("The results of the sequential and parallel computations match.\n");
    } else {
        std::printf("The results of the sequential and parallel computations do not match.\n");
    }

    return 0;
}
