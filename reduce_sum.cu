/**
 * Parallel Vector ReduceSum.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "benchmark.h"

#define SIZE (1 << 25)

// Kernels
__global__ void reduce_sum_kernel1(float *out, const float4 *arr, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    float4 val = arr[idx];
    atomicAdd(out, val.x + val.y + val.z + val.w);
  }
}

__global__ void reduce_sum_kernel2(float *out, const float *arr, int N) {
  extern __shared__ float buffer[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < N) {
    buffer[tid] = arr[idx];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2 * s) == 0) {
        buffer[tid] += buffer[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      atomicAdd(out, buffer[0]);
    }
  }
}

__global__ void reduce_sum_kernel3(float *out, const float *arr, int N) {
  extern __shared__ float buffer[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < N) {
    buffer[tid] = arr[idx];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
      int index = 2 * s * tid;
      if (index < blockDim.x) {
        buffer[index] += buffer[index + s];
      }
      __syncthreads();
    }
    
    if (tid == 0) {
      atomicAdd(out, buffer[0]);
    }
  }
}

__global__ void reduce_sum_kernel4(float *out, const float *arr, int N) {
  extern __shared__ float buffer[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < N) {
    buffer[tid] = arr[idx];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (tid < s) {
        buffer[tid] += buffer[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      atomicAdd(out, buffer[0]);
    }
  }
}

/**
 * Parallel reduction using adjacent elements (log-scaling)
 */
__device__ void warpReduce(volatile float *buffer, int tid) {
  buffer[tid] += buffer[tid + 32];
  buffer[tid] += buffer[tid + 16];
  buffer[tid] += buffer[tid + 8];
  buffer[tid] += buffer[tid + 4];
  buffer[tid] += buffer[tid + 2];
  buffer[tid] += buffer[tid + 1];
}

__global__ void reduce_sum_kernel5(float *out, float *arr, int N) {
  extern __shared__ float buffer[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < N) {
    buffer[tid] = arr[idx];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2) {
      if (tid < s) {
        buffer[tid] += buffer[tid + s];
      }
      __syncthreads();
    }

    if (tid < 32) {
      warpReduce(buffer, tid);
    }

    if (tid == 0) {
      atomicAdd(out, buffer[0]);
    }
  }
}

__global__ void reduce_sum_kernel6(float *out, float *arr, int N) {
  extern __shared__ float buffer[];
  int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  int tid = threadIdx.x;

  if (idx < N) {
    buffer[tid] = arr[idx] + arr[idx + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2) {
      if (tid < s) {
        buffer[tid] += buffer[tid + s];
      }
      __syncthreads();
    }

    if (tid < 32) {
      warpReduce(buffer, tid);
    }

    if (tid == 0) {
      atomicAdd(out, buffer[0]);
    }
  }
}

void reduce_sum(int kernel_num, float *out, float *arr, int N, int block_size) {
  int num_blocks;
  cudaCheck(cudaMemset(out, 0, sizeof(float)));

  switch (kernel_num) {
    case 1:
      num_blocks = ceil_div(N / 4, block_size);
      reduce_sum_kernel1<<<num_blocks, block_size>>>(out, (float4 *)arr, N);
      break;
    case 2:
      num_blocks = ceil_div(N, block_size);
      reduce_sum_kernel2<<<num_blocks, block_size, sizeof(float) * block_size>>>(out, arr, N);
      break;
    case 3:
      num_blocks = ceil_div(N, block_size);
      reduce_sum_kernel3<<<num_blocks, block_size, sizeof(float) * block_size>>>(out, arr, N);
      break;
    case 4:
      num_blocks = ceil_div(N, block_size);
      reduce_sum_kernel4<<<num_blocks, block_size, sizeof(float) * block_size>>>(out, arr, N);
      break;
    case 5:
      num_blocks = ceil_div(N, block_size);
      reduce_sum_kernel5<<<num_blocks, block_size, sizeof(float) * block_size>>>(out, arr, N);
      break;
    case 6:
      num_blocks = ceil_div(N / 2, block_size);
      reduce_sum_kernel5<<<num_blocks, block_size, sizeof(float) * block_size>>>(out, arr, N);
      break;
    default:
      printf("Invalid kernel number.\n");
      exit(EXIT_FAILURE);
  }
  cudaCheck(cudaGetLastError());
}

int main(int argc, char **argv) {
  int kernel_num;
  if (argc > 1) {
    kernel_num = std::atoi(argv[1]);
  }
  printf("Using kernel %d, across %d elements.\n", kernel_num, SIZE);

  // Init device.
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // Initialize vector on host
  float *x = make_random_float(SIZE);
  float sum = 0.0f;
  for (size_t i = 0; i < SIZE; i++) {
    sum += x[i];
  }

  // Copy vector to device
  float *d_x;
  float *d_o;

  cudaCheck(cudaMalloc(&d_x, sizeof(float) * SIZE));
  cudaCheck(cudaMalloc(&d_o, sizeof(float)));

  cudaCheck(cudaMemcpy(d_x, x, sizeof(float) * SIZE, cudaMemcpyHostToDevice));

  int block_sizes[] = {64, 128, 256, 512, 1024};
  for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++) {
    int block_size = block_sizes[i];
    printf("Using block size %d\n", block_size);
    reduce_sum(kernel_num, d_o, d_x, SIZE, block_size);
    validate_result(d_o, &sum, "sum", 1, 1.0f);
  }
  printf("Result verified, running benchmarks.\n");

  int repeats = 1000;
  for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++) {
    int block_size = block_sizes[i];
    float elapsed_time_ms;
    elapsed_time_ms = benchmark_kernel(repeats, reduce_sum, kernel_num, d_o, d_x, SIZE, block_size);

    long memory_ops = SIZE * 4;
    float memory_bandwidth = (float)memory_ops / elapsed_time_ms / 1e6;
    printf("block_size %4d | time %.4f | bandwidth %.4f GB/s\n", block_size, elapsed_time_ms, memory_bandwidth);
  }

  cudaCheck(cudaFree(d_x));
  cudaCheck(cudaFree(d_o));
  free(x);
  return 0;
}