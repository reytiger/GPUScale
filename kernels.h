/*********************************************
* File: kernels.cu
* Author: Kevin Carbaugh
* Colorado School of Mines, Spring 2016
* 
* Delcaraions of GPU-side functions 
*
*********************************************/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdbool.h>


// CUDA math kernel function pointer type
typedef void (*kernelPointer_t)(void*, void*, void*, int);

typedef enum {INT, FLOAT, DOUBLE} datatype_t;

// function declartions
// option setters
__global__ void set_data_type(datatype_t dt);
__global__ void set_kernel_to_run(int choice);

// math kernels
__device__ void vecadd(void *c, void *a, void *b, int gid);
__device__ void vecsub(void *c, void *a, void *b, int gid);
__device__ void vecmult(void *c, void *a, void *b, int gid);
__device__ void vecdiv(void *c, void *a, void *b, int gid);

// main driver and helpers
__device__ uint get_smid(void);

__global__ void limit_sms_kernel_shared(void *c, void *a, void *b, bool *active_sms,
        unsigned int *finished_tasks, unsigned int ntasks, unsigned int *d_wd, size_t block_size);

__global__ void limit_sms_kernel_global(void *c, void *a, void *b, bool *active_sms, 
        unsigned int *finished_tasks, unsigned int ntasks, unsigned int *d_wd, size_t block_size);
