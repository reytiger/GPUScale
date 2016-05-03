#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define BLK_SIZE 256
#define BLK_NUM 64000

#define TIME_FAIL -1.0f
//~16 million integer pairs to sum

__device__ uint get_smid(void)
{
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}


// adds a & b, stores into c using only the specified number of SMs
__global__ void add_kernel(int *c, int *a, int *b, unsigned int max_sms,
        unsigned int *finished_tasks, unsigned int ntasks)
{
  // task id is shared amongst all threads in the block
  __shared__ int taskid;
  int smid = get_smid();
  if(smid >= max_sms) 
  {
    return;
  }

  // infinite loop the thread to keep resident
  while (1)
  {
    if(threadIdx.x == 0)
    {
     taskid = atomicInc(&finished_tasks[0], INT_MAX); 
    }
    __syncthreads();

    if(taskid >= ntasks)
      return;

    int gid = taskid * BLK_SIZE + threadIdx.x;
    c[gid] = a[gid] + b[gid];
  }
}


// verifies that the GPU computed the correct results
bool verify_result(int num_elements, int *resultArray)
{
  int wrongCount = 0;
  for(int i = 0; i < num_elements; ++i)
  {
    if(resultArray[i] != 2) {
      fprintf(stderr, "Wrong result at %d : %d\n", i, resultArray[i]);
      ++wrongCount;
    }
  }

  if(wrongCount > 1)
    return false; // indicates invalid results

  return true;
}


// launches the addition kernel and records timing information
float benchmark(int* da, int* db, int* dc, int *hc, int max_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop)
{
  // reset the progress marker and result array
  cudaMemset(finished_tasks, 0, sizeof(unsigned int));
  cudaMemset(dc, 0, sizeof(int) * num_elements);

  cudaEventRecord(*start);

  // perform c = a + b
  add_kernel<<<NUM_SMS * 16, BLK_SIZE>>> (dc, da, db, max_sms, finished_tasks, BLK_NUM);

  cudaDeviceSynchronize();
  cudaEventRecord(*stop);
  cudaEventSynchronize(*stop);

  float elapsedTime = 0.f;
  cudaEventElapsedTime(&elapsedTime, *start, *stop);

  cudaMemcpy(hc, dc, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

  if(!verify_result(num_elements, hc))
    return TIME_FAIL;
  return elapsedTime;
}


// runs the benchmark serveral times to get an average
float benchmark_avg(int* da, int *db, int* dc, int* hc, int max_sms, int num_elements,
  unsigned int iterations, unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop)
{
  float total_time = 0.f;
  float elapsed_time = 0.f;
  for(unsigned int i = 0; i < iterations; ++i)
  {
    elapsed_time = benchmark(da, db, dc, hc, max_sms, num_elements, finished_tasks, start, stop);
    if(elapsed_time == TIME_FAIL)
    {
      // repeat this iteration
      --i;
      continue;
    }
    else
    {
      total_time += elapsed_time;
    }
  }

  return total_time / iterations;
}


// allocates and initializes device and host arrays
void init_arrays(int num_elements, int **hc, int **da, int **db, int **dc)
{
  // alloc host memory
  *hc = (int*)malloc(num_elements * sizeof(int));

  int *operand = (int*)malloc(sizeof(int) * num_elements);
  // init host buffers
  for(int i = 0; i < num_elements; ++i)
  {
      (*hc)[i] = 0;
      operand[i] = 1;
  }

  // alloc device memory
  cudaMalloc((void**)da, num_elements * sizeof(int));
  cudaMalloc((void**)db, num_elements * sizeof(int));
  cudaMalloc((void**)dc, num_elements * sizeof(int));

  // set operand vectors on the device
  cudaMemcpy(*da, operand, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*db, operand, num_elements * sizeof(int), cudaMemcpyHostToDevice);

  free(operand);
}


int main(int argc, char** argv)
{
  if(argc < 3)
  {
    fprintf(stderr, "Usage %s SMs_to_use iterations\n", argv[0]);
    exit(1);
  }

  unsigned int max_sms = atoi(argv[1]);
  // sound upper and lower bounds on the number of SMs active
  if(max_sms <= 0)
  {
    printf("No SMs are active, aborting.\n");
    exit(0);
  } else if(max_sms > NUM_SMS)
  {
    max_sms = NUM_SMS;
  }

  unsigned int iterations = atoi(argv[2]);

  int num_elements = BLK_SIZE * BLK_NUM;

  // host arrays
  int *hc = NULL;

  // device arrays
  int *da = NULL;
  int *db = NULL;
  int *dc = NULL;

  // one-time allocation and init
  init_arrays(num_elements, &hc, &da, &db, &dc);

  // progress counter
  unsigned int *finished_tasks;
  cudaMalloc((void**)&finished_tasks, sizeof(unsigned int));

  // setup timing events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // establish baseline time
  float baseline = benchmark_avg(da, db, dc, hc, 1, num_elements,
    iterations, finished_tasks, &start, &stop);

  printf("Average baseline time (single SM) for %i iterations: %f ms\n", iterations, baseline);

  float ntime = benchmark_avg(da, db, dc, hc, max_sms, num_elements,
    iterations, finished_tasks, &start, &stop);

  printf("Average time (%d SMs) for %i iterations: %f ms\n", max_sms, iterations, ntime);
  printf("Scalability overhead: %f%\n", (ntime / (baseline / max_sms) - 1) * 100);

  // cleanup
  free(hc);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  return 0;
}
