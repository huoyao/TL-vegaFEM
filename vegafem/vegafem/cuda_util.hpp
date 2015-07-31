#ifndef _CUDA_UTIL_HPP_
#define _CUDA_UTIL_HPP_

#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#define CUDA_ALL __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_ALL
#define CUDA_HOST
#define CUDA_DEVICE
#endif

#ifdef __CUDACC__
const int TPB = 128;
inline int bpg(const int &n,const int &tpb)	{return (n + tpb - 1) / tpb;}
#endif

inline void reportMemoryGPU()
{
  size_t free_byte;
  size_t total_byte;
  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

  if(cudaSuccess != cuda_status)
  {
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
    exit(1);
  }

  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

#endif