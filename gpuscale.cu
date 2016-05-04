#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define BLK_SIZE 128
#define BLK_NUM 65536
#define ITERATIONS 200
// 2^24 items

#define TIME_FAIL -1.0f


// CUDA kernel function pointer type
typedef void (*kernelPointer_t)(int*, int*, int*, int);


// ---All math kernels----
__device__ void vecadd(int *c, int* a, int* b, int gid)
{
  c[gid] = a[gid] + b[gid];
}


// the kernels avilable. Array resides on the device
// since the host cannot set this value directly
__device__ kernelPointer_t d_kp[] = { vecadd };


__device__ uint get_smid(void)
{
   uint ret;
   asm("mov.u32 %0, %smid;" : "=r"(ret) );
   return ret;
}


// runs a kernel on a specified number of gpus
__global__ void limit_sms_kernel(kernelPointer_t kp, int *c, int *a, int *b, unsigned int max_sms,
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
    (*kp)(c, a, b, gid);
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
float benchmark(kernelPointer_t kp, int* da, int* db, int* dc, int *hc, int max_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop)
{
  // reset the progress marker and result array
  cudaMemset(finished_tasks, 0, sizeof(unsigned int));
  cudaMemset(dc, 0, sizeof(int) * num_elements);

  cudaEventRecord(*start);

  // perform the math op
  limit_sms_kernel<<<NUM_SMS * 16, BLK_SIZE>>> (kp, dc, da, db, max_sms, finished_tasks, BLK_NUM);

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
float benchmark_avg(kernelPointer_t kp, int* da, int *db, int* dc, int* hc, int max_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, int iterations)
{
  float total_time = 0.f;
  float elapsed_time = 0.f;
  for(unsigned int i = 0; i < iterations; ++i)
  {
    elapsed_time = benchmark(kp, da, db, dc, hc, max_sms, num_elements, finished_tasks, start, stop);
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

  return total_time / ITERATIONS;
}


float establish_baseline(kernelPointer_t kp, int* da, int *db, int* dc, int* hc, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop)
{
  printf("Running %d iterations to establish baseline...\n", ITERATIONS * 3);
  // establish baseline time
  float baseline = benchmark_avg(kp, da, db, dc, hc, 1, num_elements, finished_tasks, start, stop, ITERATIONS * 3) / 3;

  printf("Average baseline time (single SM) for %i iterations: %f ms\n", ITERATIONS * 3, baseline);

  return baseline;
}


// main test driver
void run_test(kernelPointer_t kp, int* da, int *db, int* dc, int* hc, int max_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, float baseline)
{
  printf("\nRunning %d iterations on %d SMs...\n", ITERATIONS, max_sms);
  float ntime = benchmark_avg(kp, da, db, dc, hc, max_sms, num_elements,
    finished_tasks, start, stop, ITERATIONS);

  printf("Average time (%d SMs) for %i iterations: %f ms\n", max_sms, ITERATIONS, ntime);
  printf("Scalability overhead: %f%%\n", (ntime / (baseline / max_sms) - 1.0f) * 100.0f);
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
  // the number of elements to compute
  int num_elements = BLK_NUM * BLK_SIZE; //1 << 16; // 2^16 = 65536

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

  bool allSMCombos = false;

  // function pointers to the desired kernel
  kernelPointer_t h_kp;

  // main menu
  int userChoice = 0;
  for(;;)
  {
    printf("\n\n----GPU Scalability Benchmarks---------------\nSelect an option\n"
            "0 - Quit\n"
            "1 - Toggle specifing # SMs or all sucessively (currently: ");
    if(allSMCombos)
    {
      printf("Each with 1 - %d SMS)", NUM_SMS);
    }
    else
    {
      printf("Specify # per run)");
    }
    printf("\n2 - Vector Addition\n");
    printf("Please enter a choice: ");
    scanf(" %d" , &userChoice);

    int kernelIdx = 0;
    switch(userChoice)
    {
      case 0:
        // cleanup
        free(hc);

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        return 0;
      case 1:
        allSMCombos = !allSMCombos;
        continue;
      case 2:
        // TODO add more cases
        kernelIdx = userChoice - 2;
        break;
      default:
        printf("Could not recognize your choice\n");
        // retry menu
        continue;
    }

    // get a pointer to the device function into host memory
    cudaMemcpyFromSymbol(&h_kp, d_kp[kernelIdx], sizeof(kernelPointer_t));

    float baseline = 0.f;
    // run the indicated test
    if(allSMCombos)
    {
      baseline = establish_baseline(h_kp, da, db, dc, hc, num_elements, finished_tasks, &start, &stop);
      for(int sms_count = 1; sms_count <= NUM_SMS; ++sms_count)
      {
        run_test(h_kp, da, db, dc, hc, sms_count, num_elements, finished_tasks, &start, &stop, baseline);
      }
    }
    else
    {
      int sms_choice = -1;
      while(sms_choice < 0 || sms_choice > NUM_SMS)
      {
        printf("\nHow many SMs would you like to run this kernel on?: ");
        scanf(" %d", &sms_choice);
      }

      baseline = establish_baseline(h_kp, da, db, dc, hc, num_elements, finished_tasks, &start, &stop);
      run_test(h_kp, da, db, dc, hc, sms_choice, num_elements, finished_tasks, &start, &stop, baseline);
    }
  }
  // return 0;
}
