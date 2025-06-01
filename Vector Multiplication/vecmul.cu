#include "solve.h"
#include <cuda_runtime.h>
#include <assert.h>

// #define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    /*
    so invocation in computation, is like grid, block, thread
    so threads in a block in a grid
    now, threads that are in same block, have access to shared memeory region of that block, so each block have their
    own SMEM (share memory), 

    for each run, we can configure number of threads in a block but again maximum it can take 1024 threads
    this is configured via blockDim (block dimension): i.e x, y, z (say l, b, h in 3d ) but is called, x, y, z

    so to get offset, i.e how far we are in the block, for computation, 
    
    since in normal matrix, we do dot product, for say first element in final result matrix, we do dot product of 
    first row and first column

    we can do each thread and each entry in result matrix same, like 1:1 mapping kind of, this way there is no
    race condition and no need to synchronize.

    sgemm: C= alpha * A * B  + beta *C

    here a, b, c (a*b) are matrix 
    alpha and beta: scalers in single point precision float 
    alpha is used to scale A * B
    beta is to scale C

    now scale means:  like changing a number, by multiply (it can different operation though)
    i.e say scale 5 by 2 is 2 * 5

    for above we can make alpha as 1.0 and beta as 0.0, so the final result is C
    now, why would we want to use scale is, like it can final result,
    like scale down final result by say double the multiple and then scled it by half, if alpha is 2.0 and beta is 0.5

    reason is, there are many implimentation of matrix multiplication, where the weights are used to influence the final result
    it does not matter if we just get the normal multiplication result, say we use it in 
    learning machine alogrithm, (just arrange it correct order while reading, its tought to write notes you know ;))

    Below solution still fails at matrix having different dimensions like m*n, n*k, 
    */
    int row = blockIdx.y * blockDim.y  +  threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        float sum = 0;
        for(int i = 0; i < N;  i++){
            sum += A[row*N + i] * B[i*K + col];
        }
        C[row*K + col] = sum;
    }
}

// __global__
// void naive_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
//     // get the row and col values using block, thread, idx and dim
//     const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint y = blockIdx.y * blockDim.x + threadIdx.y;

//     // just check if we cross the thread range, to avoid index out of loop
//     if (x < M && y < N){
//         float tmp = 0.0; // this will be used with alpha, like A[i] * B [i] * alpha
//         for (int i = 0; i< K ; i++){
//             tmp += A[x*K + i] * B[x*N  + y];
//         }
//         C[x*N + y] = alpha*tmp + beta * C[x * N + y];
//     }
// }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    // a naive multiplication with each thread to each entry in result matrix
    // this is to make blocks, needed to map all elements of C
    // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // just maximum threads possible in a block: 32*32
    // dim3 blockDim(32, 32, 1);
    // launching the kernel
    // naive_matmul<<<gridDim, blockDim>>> (M, N, K, 1.0, A, B, 0.0 , C);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
