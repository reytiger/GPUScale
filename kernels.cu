/*********************************************
* File: kernels.cu
* Author: Kevin Carbaugh
* Colorado School of Mines, Spring 2016
* 
* Implementation of GPU-side functions 
*
*********************************************/

#include "kernels.h"

// a pointer to the math kernel to execute
__device__ kernelPointer_t kp = vecadd;

// the data type to use for math functions
__device__ datatype_t data_type = INT;


// ---All math kernels----
__device__ void vecadd(void *c, void *a, void *b, int gid)
{
  switch(data_type)
  {
  case INT:
    ((int*)c)[gid] = ((int*)a)[gid] + ((int*)b)[gid];
    break;
  case FLOAT:
    ((float*)c)[gid] = ((float*)a)[gid] + ((float*)b)[gid];
    break;
  case DOUBLE:
    ((double*)c)[gid] = ((double*)a)[gid] + ((double*)b)[gid];
    break;
  }
}


__device__ void vecsub(void *c, void *a, void *b, int gid)
{
  switch(data_type)
  {
  case INT:
    ((int*)c)[gid] = ((int*)a)[gid] - ((int*)b)[gid];
    break;
  case FLOAT:
    ((float*)c)[gid] = ((float*)a)[gid] - ((float*)b)[gid];
    break;
  case DOUBLE:
    ((double*)c)[gid] = ((double*)a)[gid] - ((double*)b)[gid];
    break;
  }
}


__device__ void vecmult(void *c, void *a, void *b, int gid)
{
  switch(data_type)
  {
  case INT:
    ((int*)c)[gid] = ((int*)a)[gid] * ((int*)b)[gid];
    break;
  case FLOAT:
    ((float*)c)[gid] = ((float*)a)[gid] * ((float*)b)[gid];
    break;
  case DOUBLE:
    ((double*)c)[gid] = ((double*)a)[gid] * ((double*)b)[gid];
    break;
  }
}


__device__ void vecdiv(void *c, void *a, void *b, int gid)
{
  switch(data_type)
  {
  case INT:
    ((int*)c)[gid] = ((int*)a)[gid] / ((int*)b)[gid];
    break;
  case FLOAT:
    ((float*)c)[gid] = ((float*)a)[gid] / ((float*)b)[gid];
    break;
  case DOUBLE:
    ((double*)c)[gid] = ((double*)a)[gid] / ((double*)b)[gid];
    break;
  }
}


__global__ void set_data_type(datatype_t dt)
{
  data_type = dt;
}

// helper function to change which math op is run
__global__ void set_kernel_to_run(int choice)
{
  switch(choice)
  {
  case 0:
    kp = vecadd;
    break;
  case 1:
    kp = vecsub;
    break;
  case 2:
    kp = vecmult;
    break;
  case 3:
    kp = vecdiv;
    break;
  }
}

__device__ uint get_smid(void)
{
   uint ret;
   asm("mov.u32 %0, %smid;" : "=r"(ret) );
   return ret;
}


// runs a kernel on a specified number of gpus using args that are first buffered in L2 shared memory
__global__ void limit_sms_kernel_shared(void *c, void *a, void *b, bool *active_sms,
        unsigned int *finished_tasks, unsigned int ntasks, unsigned int *d_wd, size_t block_size)
{
  // task id is shared amongst all threads in the block
  __shared__ int taskid;
  int smid = get_smid();
  if(!active_sms[smid])
  {
    return;
  }

  // the two operands and results storage space, as one matrix

  extern __shared__ char L2[];

  void *operand1 = L2;
  void *operand2;
  void *result;
  switch(data_type)
  {
  case INT:
    operand2 = &((int*)operand1)[block_size];
    result = &((int*)operand2)[block_size];
    break;
  case FLOAT:
    operand2 = &((float*)operand1)[block_size];
    result = &((float*)operand2)[block_size];
    break;
  case DOUBLE:
    operand2 = &((double*)operand1)[block_size];
    result = &((double*)operand2)[block_size];
    break;
  }

  // infinite loop the thread to keep resident
  while (1)
  {
    if(threadIdx.x == 0)
    {
      taskid = atomicInc(&finished_tasks[0], INT_MAX); 

      // record the work being done by this block on the SM
      if(d_wd && taskid <= ntasks)
      {
        atomicInc(&d_wd[smid], INT_MAX);
      }
    }
    __syncthreads();

    if(taskid >= ntasks)
      return;
    
    int sumIdx = taskid * block_size + threadIdx.x;

    // load from global to shared
    switch(data_type)
    {
    case INT:
      ((int*)operand1)[threadIdx.x] = ((int*)a)[sumIdx];
      ((int*)operand2)[threadIdx.x] = ((int*)b)[sumIdx];
      break;
    case FLOAT:
      ((float*)operand1)[threadIdx.x] = ((float*)a)[sumIdx];
      ((float*)operand2)[threadIdx.x] = ((float*)b)[sumIdx];
      break;
    case DOUBLE:
      ((double*)operand1)[threadIdx.x] = ((double*)a)[sumIdx];
      ((double*)operand2)[threadIdx.x] = ((double*)b)[sumIdx];
      break;
    }
    __syncthreads();

    // launch the kernel using shared memory
    (*kp)(result, operand1, operand2, threadIdx.x);

    // copy result from shared back to global
    switch(data_type)
    {
    case INT:
      ((int*)c)[sumIdx] = ((int*)result)[threadIdx.x];
      break;
    case FLOAT:
      ((float*)c)[sumIdx] = ((float*)result)[threadIdx.x];
      break;
    case DOUBLE:
      ((double*)c)[sumIdx] = ((double*)result)[threadIdx.x];
      break;
    }

  }
}


// runs a kernel on a specified number of gpus using args in global memory
__global__ void limit_sms_kernel_global(void *c, void *a, void *b, bool *active_sms, 
        unsigned int *finished_tasks, unsigned int ntasks, unsigned int *d_wd, size_t block_size)
{
  // task id is shared amongst all threads in the block
  __shared__ int taskid;
  int smid = get_smid();
  if(!active_sms[smid])
  {
    return;
  }

  // infinite loop the thread to keep resident
  while (1)
  {
    if(threadIdx.x == 0)
    {
      taskid = atomicInc(&finished_tasks[0], INT_MAX); 

      // record the work being done by this block on the SM
      if(d_wd && taskid <= ntasks)
      {
        atomicInc(&d_wd[smid], INT_MAX);
      }
    }
    __syncthreads();

    if(taskid >= ntasks)
      return;

    // launch the kernel
    (*kp)(c, a, b, taskid * block_size + threadIdx.x);
  }
}
