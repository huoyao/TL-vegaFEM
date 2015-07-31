//MDF BY TL
#ifndef _ATOMICOP_CUH_
#define _ATOMICOP_CUH_

__device__ double atomicAdd(double* address, double val);

inline __device__  void atomicAdd(double3 *address, const double3 &val);

#endif