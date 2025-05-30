#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    /*
    Notes:
        gridDim.x is like blocks count in a grid
        blockIdx.x is like index of current thread in the block in the grid
        
        so to get the thread index of any thread (like starting from 0)
        we need to use the offset i.e how fat we are from starting and then the index plus
        so : blockIdx.x  (thisi the index or like which block we are in ) * 
             blockDim.x (this is the dimension of each block)  
            (these two to get how far we are in grid) and 
        then add threadIdx.x (current thread index
        now to find count/add all threads in the grid, we need to keep adding all threads present in a grid
        which is again blockDim.x * gridDim.x (this is like in a grid,take below example:
        like each block can have 1024 threads, which is product of its dimension (l * b * h)
        maximum dimensions can be 1024, 1024, 64 (l, b , h) , 
        so stride specially grid stride loop is to access data efficiently when data size is like too big and more
        than the maximum thread count in a grid. like divide and then keep jumping that stride length. :
        which will be 
        blockDim.x => number of threads in a block
        gridDim.x => number of blocks in a grid
    */
    int start = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i=start; i<N; i+=stride){
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
