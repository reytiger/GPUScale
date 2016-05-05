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


__device__ void vecsub(int *c, int* a, int* b, int gid)
{
  c[gid] = a[gid] - b[gid];
}


__device__ void vecmult(int *c, int* a, int* b, int gid)
{
  c[gid] = a[gid] * b[gid];
}


__device__ void vecdiv(int *c, int* a, int* b, int gid)
{
  c[gid] = a[gid] / b[gid];
}

// the kernels avilable. Array resides on the device
// since the host cannot set this value directly
__device__ kernelPointer_t d_kp[] = { vecadd, vecsub, vecmult, vecdiv };


__device__ uint get_smid(void)
{
   uint ret;
   asm("mov.u32 %0, %smid;" : "=r"(ret) );
   return ret;
}


// runs a kernel on a specified number of gpus using args that are first buffered in L2 shared memory
__global__ void limit_sms_kernel_shared(kernelPointer_t kp, int *c, int *a, int *b, unsigned int max_sms,
        unsigned int *finished_tasks, unsigned int ntasks, unsigned int *d_wd)
{
  // task id is shared amongst all threads in the block
  __shared__ int taskid;
  int smid = get_smid();
  if(smid >= max_sms) 
  {
    return;
  }

  // the two operands and results storage space, as one matrix
  extern __shared__ int L2[];

  // split into three even sized arrays
  // size_t sizePerOp = SHARED_SIZE / NUM_SMS / 16 / 3;
  size_t sizePerOp = BLK_SIZE;
  int *operand1 = L2;
  int *operand2 = &operand1[sizePerOp];
  int *result = &operand2[sizePerOp];

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
    
    int sumIdx = taskid * BLK_SIZE + threadIdx.x;

    // load from global to shared
    operand1[threadIdx.x] = a[sumIdx];
    operand2[threadIdx.x] = b[sumIdx];
    __syncthreads();

    // launch the kernel using shared memory
    (*kp)(result, operand1, operand2, threadIdx.x);

    // copy result from shared back to global
    c[sumIdx] = result[threadIdx.x];

  }
}


// runs a kernel on a specified number of gpus using args in global memory
__global__ void limit_sms_kernel_global(kernelPointer_t kp, int *c, int *a, int *b, unsigned int max_sms,
        unsigned int *finished_tasks, unsigned int ntasks, unsigned int *d_wd)
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
    (*kp)(c, a, b, taskid * BLK_SIZE + threadIdx.x);
  }
}


// verifies that the GPU computed the correct results
bool verify_result(int num_elements, int *resultArray, int expected)
{
  int wrongCount = 0;
  for(int i = 0; i < num_elements; ++i)
  {
    if(resultArray[i] != expected) {
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
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, unsigned int *d_wd, bool shared_mem, int expected)
{
  // reset the progress marker and result array
  cudaMemset(finished_tasks, 0, sizeof(unsigned int));
  cudaMemset(dc, 0, sizeof(int) * num_elements);

  cudaEventRecord(*start);

  // perform the math op
  if(shared_mem)
  {
    //size_t shared_per_block = SHARED_SIZE / NUM_SMS / 16;
    limit_sms_kernel_shared<<<NUM_SMS * 16, BLK_SIZE, BLK_SIZE * 3 * sizeof(int)>>> (kp, dc, da, db, max_sms, finished_tasks, BLK_NUM, d_wd);
  }
  else
  {
    limit_sms_kernel_global<<<NUM_SMS * 16, BLK_SIZE>>> (kp, dc, da, db, max_sms, finished_tasks, BLK_NUM, d_wd);
  }

  cudaDeviceSynchronize();
  cudaEventRecord(*stop);
  cudaEventSynchronize(*stop);

  float elapsedTime = 0.f;
  cudaEventElapsedTime(&elapsedTime, *start, *stop);

  cudaMemcpy(hc, dc, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

  if(!verify_result(num_elements, hc, expected))
    return TIME_FAIL;
  return elapsedTime;
}


// runs the benchmark serveral times to get an average
float benchmark_avg(kernelPointer_t kp, int* da, int *db, int* dc, int* hc, int max_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, int iterations, unsigned int *d_wd, bool shared_mem, int expected)
{
  float total_time = 0.f;
  float elapsed_time = 0.f;

  if(d_wd)
    // reset work distribution counters
    cudaMemset(d_wd, 0, sizeof(int) * NUM_SMS);

  for(unsigned int i = 0; i < iterations; ++i)
  {
    elapsed_time = benchmark(kp, da, db, dc, hc, max_sms, num_elements, finished_tasks, start, stop, d_wd, shared_mem, expected);
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
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, bool shared_mem, int expected)
{
  printf("Running %d iterations over %d elements to establish baseline...\n", ITERATIONS * 3, num_elements);
  // establish baseline time
  float baseline = benchmark_avg(kp, da, db, dc, hc, 1, num_elements, finished_tasks, start, stop, ITERATIONS * 3, NULL, shared_mem, expected) / 3;

  printf("Average baseline time (single SM) for %i iterations: %f ms\n", ITERATIONS * 3, baseline);

  return baseline;
}


// main test driver
void run_test(kernelPointer_t kp, int* da, int *db, int* dc, int* hc, int max_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, float baseline,
  unsigned int* h_wd, unsigned int *d_wd, bool shared_mem, int expected)
{
  printf("\nRunning %d iterations on %d SMs...\n", ITERATIONS, max_sms);
  float ntime = 0.f;
  
  // a single SMs is equal to the baseline
  if(max_sms > 1)
  {
    ntime = benchmark_avg(kp, da, db, dc, hc, max_sms, num_elements,
      finished_tasks, start, stop, ITERATIONS, d_wd, shared_mem, expected);
  }
  else
  {
    ntime = baseline;
  }

  printf("Average time (%d SMs) for %i iterations: %f ms\n", max_sms, ITERATIONS, ntime);
  printf("Scalability overhead: %f%%\n", (ntime / (baseline / max_sms) - 1.0f) * 100.0f);

  // get the work distribution stats
  if(d_wd)
  {
    cudaMemcpy(h_wd, d_wd, NUM_SMS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Work distribution:\n");

    unsigned int total_work = 0;
    for(int i = 0; i < NUM_SMS; ++i)
      total_work += h_wd[i] / ITERATIONS;

    printf("Total thread block work units: %u\n", total_work);
    for(int i = 0; i < NUM_SMS; ++i)
    {
      // skip unused SMs
      if(h_wd[i] == 0)
        continue;

      printf("  SM #%d - %u tasks\n", i, h_wd[i] / ITERATIONS);
      float percentage = (h_wd[i] / ITERATIONS / (float)total_work) * 100.f;
      printf("  SM #%d - %f%%\n", i, percentage);
    }
  }
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
  unsigned int h_wd[NUM_SMS];

  // device arrays
  int *da = NULL;
  int *db = NULL;
  int *dc = NULL;
  unsigned int *d_wd = NULL;

  // progress counter
  unsigned int *finished_tasks;

  // one-time allocation and init
  init_arrays(num_elements, &hc, &da, &db, &dc);
  cudaMalloc((void**)&d_wd, sizeof(unsigned int) * NUM_SMS);
  cudaMalloc((void**)&finished_tasks, sizeof(unsigned int));

  // setup timing events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  bool allSMCombos = false;
  bool collectDist = false;
  bool useSharedMem = false;

  // function pointers to the desired kernel
  kernelPointer_t h_kp;

  // main menu
  int userChoice = 0;
  for(;;)
  {
    printf("\n\n----GPU Scalability Benchmarks---------------\nSelect an option\n"
            "0 - Quit\n"
            "--OPTIONS--\n"
            "1 - Toggle specifing # SMs or all sucessively (currently: ");
    if(allSMCombos)
    {
      printf("Each with 1 - %d SMS)", NUM_SMS);
    }
    else
    {
      printf("Specify # per run)");
    }

    printf("\n2 - Toggle work distribution collection & display (currently: ");
    if(collectDist)
    {
      printf("ON)");
    }
    else
    {
      printf("OFF)");
    }

    printf("\n3 - Toggle using shared L2 cache or global memory (currently: ");
    if(useSharedMem)
    {
      printf("Shared)");
    }
    else
    {
      printf("Global)");
    }
    printf("\n--KERNELS--\n"
        "4 - Vector Addition\n"
        "5 - Vector Subtraction\n"
        "6 - Vector Multiplication\n"
        "7 - Vector Division\n");
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
        cudaFree(d_wd);
        return 0;
      case 1:
        allSMCombos = !allSMCombos;
        continue;
      case 2:
        collectDist = !collectDist;
        continue;
      case 3:
        useSharedMem = !useSharedMem;
        continue;
      case 4:
      case 5:
      case 6:
      case 7:
        kernelIdx = userChoice - 4;
        break;

      default:
        printf("Could not recognize your choice\n");
        // retry menu
        continue;
    }

    // get a pointer to the device function into host memory
    cudaMemcpyFromSymbol(&h_kp, d_kp[kernelIdx], sizeof(kernelPointer_t));


    // figure out what result to expect based on which kernel was invoked
    int expectedValue;
    switch(userChoice - 4)
    {
    case 0: //add
      expectedValue = 2;
      break;
    case 1: //sub
      expectedValue = 0;
      break;
    case 2: //mult
    case 3: //div
      expectedValue = 1;
      break;
    default:
      fprintf(stderr, "Error! Could not determine kernel specified!");
      exit(2);
    }

    unsigned int *workDist = NULL;
    if(collectDist)
      workDist = d_wd;

    float baseline = 0.f;
    // run the indicated test
    if(allSMCombos)
    {
      baseline = establish_baseline(h_kp, da, db, dc, hc, num_elements, finished_tasks, &start, &stop, useSharedMem, expectedValue);
      for(int sms_count = 2; sms_count <= NUM_SMS; ++sms_count)
      {
        run_test(h_kp, da, db, dc, hc, sms_count, num_elements, finished_tasks,
            &start, &stop, baseline, h_wd, workDist, useSharedMem, expectedValue);
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

      baseline = establish_baseline(h_kp, da, db, dc, hc, num_elements, finished_tasks, &start, &stop, useSharedMem, expectedValue);
      run_test(h_kp, da, db, dc, hc, sms_choice, num_elements, finished_tasks,
          &start, &stop, baseline, h_wd, workDist, useSharedMem, expectedValue);
    }
  }
}
