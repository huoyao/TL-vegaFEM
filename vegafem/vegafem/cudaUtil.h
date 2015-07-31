//2015-4-17 09:57:02
#ifndef _CUDAUTIL_H_
#define _CUDAUTIL_H_
#include "cuda_runtime.h"

__global__ void kernel_dotP_reduction_start(const int n, double *r, double *invDiagonal, double *for_share, bool second);

__global__ void kernel_dotP_reduction_end(double *for_share);

__global__ void kernel_SPMV(const int n,const int *row,const int *col,const double *val,const double *x,double *r);
#endif