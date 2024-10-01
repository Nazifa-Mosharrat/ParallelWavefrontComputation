#!/bin/bash
#SBATCH --job-name=wavefront_calculation   # Job name
#SBATCH --output=wavefront_output.txt   # Output file name with job ID
#SBATCH --error=wavefront_error.txt     # Error file name with job ID
#SBATCH --nodes=4                          # Number of nodes
#SBATCH --ntasks=4                 	   # Total number of MPI tasks




# Compile the MPI-based program
mpicxx --version
make wavefrontMPI

# Parameters for running the MPI program
N=1000                              # Size of the matrix
min=0                               # Minimum random value
max=1000                            # Maximum random value

# Run the MPI program
echo "========== Wavefront computation via MPI =========="
mpirun -n $SLURM_NTASKS ./wavefrontMPI $N $min $max

# Compile the FastFlow-based program
echo "========== Wavefront computation via FastFlow =========="
g++ -std=c++20 -I./include -I/home/n.mosharrat/SPMProject1/fastflow/fastflow-master -DNO_DEFAULT_MAPPING wavefrontFastflow.cpp -o wavefrontFastflow

# Run the FastFlow-based wavefront computation program
./wavefrontFastflow
