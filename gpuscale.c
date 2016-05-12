/*********************************************
* File: gpuscale.c
* Author: Kevin Carbaugh
* Colorado School of Mines, Spring 2016
* 
* Contains menu system benchmark selection logic
*
*********************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "kernels.h"

#define BLK_SIZE 128
#define BLK_NUM 65536
#define ITERATIONS 200
// 2^23 items

#define TIME_FAIL -1.0f

typedef enum {ALL_SM_COMBOS = 0, COLLECT_LOAD_DIST, USE_SHARED, WRITE_REPORT} user_options;
static const size_t OPTION_COUNT = 4;

const char *kernels[] = {"addition", "subtraction", "multiplication", "division"};

// verifies that the GPU computed the correct results
bool verify_result(int num_elements, void *result_array, double expected_value, datatype_t dt)
{
  int wrong_count = 0;

  // stores the first wrong value for display
  int wrong_int = 0;
  float wrong_float = 0.f;
  double wrong_double = 0.0;

  for(int i = 0; i < num_elements; ++i)
  {
    // expected value is alwasy double, should promote ints and floats

    bool correct_result = false;
    switch(dt)
    {
    case INT:
      correct_result = ((int *)result_array)[i] == (int)expected_value;
      break;
    case FLOAT:
      correct_result = ((float *)result_array)[i] == (float)expected_value;
      break;
    case DOUBLE:
      correct_result = ((double *)result_array)[i] == expected_value;
      break;
    }

    if(!correct_result)
    {
      ++wrong_count;

      if(wrong_count == 0)
      {
        switch(dt)
        {
        case INT:
          //fprintf(stderr, "Wrong result at %d : %d\n", i, ((int *)result_array)[i]);
          wrong_int = ((int*)result_array)[i];
          break;
        case FLOAT:
          //fprintf(stderr, "Wrong result at %d : %f\n", i, ((float *)result_array)[i]);
          wrong_float = ((float*)result_array)[i];
          break;
        case DOUBLE:
          //fprintf(stderr, "Wrong result at %d : %f\n", i, ((double *)result_array)[i]);
          wrong_double = ((double*)result_array)[i];
          break;
        }
      }
    }
  }

  if(wrong_count > 1)
  {
    // display a digest of the errors
    switch(dt)
    {
    case INT:
      fprintf(stderr, "%d wrong results! Expected %d, but computed value: %d\n", wrong_count, (int)expected_value, wrong_int);
      break;
    case FLOAT:
      fprintf(stderr, "%d wrong results! Expected %f, but computed value: %f\n", wrong_count, (float)expected_value, wrong_float);
      break;
    case DOUBLE:
      fprintf(stderr, "%d wrong results! Expected %f, but computed value: %f\n", wrong_count, (double)expected_value, wrong_double);
      break;
    }

    return false; // indicates invalid results
  }

  return true;
}


// launches the addition kernel and records timing information
float benchmark(void *da, void *db, void *dc, void *hc, bool *active_sms, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t *start, cudaEvent_t *stop, unsigned int *d_wd,
  bool *options, double expected_result, datatype_t dt)
{
  // reset the progress marker
  cudaMemset(finished_tasks, 0, sizeof(unsigned int));

  cudaEventRecord(*start);

  size_t data_size = 0;
  switch(dt)
  {
  case INT:
    data_size = sizeof(int);
    break;
  case FLOAT:
    data_size = sizeof(float);
    break;
  case DOUBLE:
    data_size = sizeof(double);
    break;
  }

  // reset the result array
  cudaMemset(dc, 0, data_size * num_elements);

  // perform the math op
  if(options[USE_SHARED])
  {
    // allocate 3 data elements for each thread in each block
    limit_sms_kernel_shared<<<NUM_SMS * 16, BLK_SIZE, BLK_SIZE * 3 * data_size>>> (dc, da, db, active_sms,
      finished_tasks, BLK_NUM, d_wd, BLK_SIZE);
  }
  else
  {
    for(int i = 0; i < 100; ++i)
    {
      limit_sms_kernel_global<<<NUM_SMS * 16, BLK_SIZE>>> (dc, da, db, active_sms,
        finished_tasks, BLK_NUM, d_wd, BLK_SIZE);

      // reset the progress marker
      cudaMemset(finished_tasks, 0, sizeof(unsigned int));
    }
  }

  // sync with the device
  cudaDeviceSynchronize();
  cudaEventRecord(*stop);
  cudaEventSynchronize(*stop);

  float elapsedTime = 0.f;
  cudaEventElapsedTime(&elapsedTime, *start, *stop);

  cudaMemcpy(hc, dc, num_elements * data_size, cudaMemcpyDeviceToHost);

  if(!verify_result(num_elements, hc, expected_result, dt))
    return TIME_FAIL;
  return elapsedTime;
}


// runs the benchmark serveral times to get an average
float benchmark_avg(void* da, void *db, void* dc, void* hc, int active_sm_count, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, int iterations,
  unsigned int *d_wd, bool *options, double expected_result, datatype_t dt)
{
  float total_time = 0.f;
  float elapsed_time = 0.f;

  if(d_wd)
    // reset work distribution counters
    cudaMemset(d_wd, 0, sizeof(unsigned int) * NUM_SMS);

  // seed the rng
  srand(time(NULL));
  for(unsigned int i = 0; i < iterations; ++i)
  {
    //printf("Picking SMs to run on for iteration #%d...", i);
    // randomly pick SMs to run on
    bool active_sms[NUM_SMS];
    for(int j = 0; j < NUM_SMS; ++j)
    {
      if(active_sm_count == NUM_SMS)
        active_sms[j] = true;
      else
        active_sms[j] = false;
    }
    
    if(active_sm_count != NUM_SMS)
    {
      int num_picked = 0;
      while(num_picked < active_sm_count)
      {
        int idx = rand() % NUM_SMS;
        if(active_sms[idx] == false)
        {
          //printf("Picked SM #%d\n", idx);
          active_sms[idx] = true;
          ++num_picked;
        }
      }
    }

    // copy to the device
    bool *d_active_sms;
    cudaMalloc((void **)&d_active_sms, NUM_SMS * sizeof(bool));
    cudaMemcpy(d_active_sms, active_sms, NUM_SMS * sizeof(bool), cudaMemcpyHostToDevice);

    elapsed_time = benchmark(da, db, dc, hc, d_active_sms, num_elements, finished_tasks, start, stop, d_wd, options, expected_result, dt);
    if(elapsed_time == TIME_FAIL)
    {
      printf("Iteration #%d had errors, retrying... Press enter to continue\n", i);
      getchar();
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


float establish_baseline(void* da, void *db, void* dc, void* hc, int num_elements,
  unsigned int *finished_tasks, cudaEvent_t* start, cudaEvent_t* stop, bool *options, double expected_result, datatype_t dt)
{
  printf("Running %d iterations over %d elements to establish baseline...\n", ITERATIONS * 3, num_elements);
  // establish baseline time
  float baseline = benchmark_avg(da, db, dc, hc, 1, num_elements, finished_tasks, start, stop, ITERATIONS * 3, NULL, options, expected_result, dt) / 3;

  printf("Average baseline time (single SM) for %i iterations: %f ms\n", ITERATIONS * 3, baseline);

  return baseline;
}


void print_results(int active_sm_count, float baseline, float runtime, unsigned int *d_wd, FILE *results_file)
{
  float overhead = (runtime / (baseline / active_sm_count) - 1.0f) * 100.0f;
  printf("Average time (%d SMs) for %i iterations: %f ms\n", active_sm_count, ITERATIONS, runtime);
  printf("Scalability overhead: %f%%\n", overhead);
  if(results_file != NULL)
  {
    fprintf(results_file, "%02d\t\t%f\t%f\n", active_sm_count, runtime, overhead);
  }

  // get the work distribution stats
  if(d_wd)
  {
    unsigned int h_wd[NUM_SMS];
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
  printf("\n\n");
}


// allocates and initializes device and host arrays with a given datatype
void alloc_with_datatype(int num_elements, void **hc, void **da, void **db, void **dc, datatype_t dt)
{
  size_t data_size = 0;
  switch(dt)
  {
  case INT:
    data_size = sizeof(int);
    break;
  case FLOAT:
    data_size = sizeof(float);
    break;
  case DOUBLE:
    data_size = sizeof(double);
    break;
  }

  // alloc host memory
  *hc = malloc(num_elements * data_size);
  // temp host-size array to copy operands to device
  void *operand = malloc(num_elements * data_size);

  // init host buffers
  for(int i = 0; i < num_elements; ++i)
  {
    switch(dt)
    {
    case INT:
      ((int*)(*hc))[i] = 0;
      ((int*)operand)[i] = 1;
      break;
    case FLOAT:
      ((float*)(*hc))[i] = 0.f;
      ((float*)operand)[i] = 1.f;
      break;
    case DOUBLE:
      ((double*)(*hc))[i] = 0.0;
      ((double*)operand)[i] = 1.0;
      break;
    }
  }

  // alloc device memory
  cudaMalloc(da, num_elements * data_size);
  cudaMalloc(db, num_elements * data_size);
  cudaMalloc(dc, num_elements * data_size);

  // set operand vectors on the device
  cudaMemcpy(*da, operand, num_elements * data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(*db, operand, num_elements * data_size, cudaMemcpyHostToDevice);

  free(operand);
}


// main test driver
void run_test(int num_elements, bool *options, datatype_t dt, double expected_result)
{

  printf("Allocating operands and result arrays...\n");
  // a global value of the number of tasks completed
  unsigned int *finished_tasks;
  cudaMalloc((void**)&finished_tasks, sizeof(unsigned int));

  // setup timing events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // pointers to the operands and results arrays
  // hc = host results
  // da = device first operand
  // db = device second operand
  // dc = device result
  // d_wd = device work distribution counter
  void *hc, *da, *db, *dc;

  unsigned int *d_wd = NULL;
  if(options[COLLECT_LOAD_DIST])
    cudaMalloc((void**)&d_wd, sizeof(unsigned int) * NUM_SMS);
  
  alloc_with_datatype(num_elements, &hc, &da, &db, &dc, dt);

  int active_sm_count = 0;
  if(!options[ALL_SM_COMBOS])
  {
    while(active_sm_count <= 0 || active_sm_count > NUM_SMS)
    {
      printf("\nHow many SMs would you like to run this kernel on?: ");
      scanf(" %d", &active_sm_count);
    }
  }

  // optional file to write results into
  FILE *results_file = NULL;
  if(options[WRITE_REPORT])
  {
    // open a file to store the results in
    char *filename;
    printf("Enter a filename to save the test results under: ");
    scanf("%ms", &filename);
    results_file = fopen(filename, "w");
    free(filename);

    if(results_file == NULL)
    {
      fprintf(stderr, "Error! Could not open results.dat for writing!\n");
      return;
    }
    // write the file header
    fprintf(results_file, "Active SMs\tRuntime (ms)\tOverhead (%%)\n");
  }

  float baseline = establish_baseline(da, db, dc, hc, num_elements, finished_tasks,
                                      &start, &stop, options, expected_result, dt);

  if(options[WRITE_REPORT])
  {
    // write the first line of the file
    fprintf(results_file,"01\t\t%f\t0\n", baseline);
  }

  float runtime = 0.f;
  // run one or many times?
  if(options[ALL_SM_COMBOS])
  {
    printf("\nRunning %d iterations on 1 - %d SMs...\n", ITERATIONS, NUM_SMS);
    for(active_sm_count = 2; active_sm_count <= NUM_SMS; ++active_sm_count)
    {
      runtime = benchmark_avg(da, db, dc, hc, active_sm_count, num_elements,
        finished_tasks, &start, &stop, ITERATIONS, d_wd, options, expected_result, dt);
      print_results(active_sm_count, baseline, runtime, d_wd, results_file);
    }
  }
  else
  {
    if(active_sm_count > 1)
    {
      printf("\nRunning %d iterations on %d SMs...\n", ITERATIONS, active_sm_count);
      runtime = benchmark_avg(da, db, dc, hc, active_sm_count, num_elements,
        finished_tasks, &start, &stop, ITERATIONS, d_wd, options, expected_result, dt);
      print_results(active_sm_count, baseline, runtime, d_wd, results_file);
    }
    else
    {
      // a single SM is equal to the baseline
      print_results(active_sm_count, baseline, baseline, d_wd, results_file);
    }
  }

  // cleanup
  if(options[WRITE_REPORT])
  {
    fclose(results_file);
  }
  free(hc);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(d_wd);
}


void set_options(bool *options, datatype_t *dt)
{
  int user_choice = 0;
  for(;;)
  {
    printf("\n--OPTIONS--\n"
           "0 - Back to main menu\n"
           "1 - Toggle specifing # SMs or all sucessively (currently: ");
    if(options[ALL_SM_COMBOS])
    {
      printf("Each with 1 - %d SMS)", NUM_SMS);
    }
    else
    {
      printf("Specify # per run)");
    }

    printf("\n2 - Toggle work distribution collection & display (currently: ");
    if(options[COLLECT_LOAD_DIST])
    {
      printf("ON)");
    }
    else
    {
      printf("OFF)");
    }

    printf("\n3 - Toggle using shared L2 cache or global memory (currently: ");
    if(options[USE_SHARED])
    {
      printf("Shared)");
    }
    else
    {
      printf("Global)");
    }
    printf("\n4 - Toggle writing results to file (Currently: ");

    if(options[WRITE_REPORT])
    {
      printf("Writing to file)");
    }
    else
    {
      printf("Terminal display only)");
    }

    printf("\n5 - Change operand & result data type (currently: ");
    switch(*dt)
    {
    case INT:
      printf("Integer)");
      break;
    case FLOAT:
      printf("Single precision float)");
      break;
    case DOUBLE:
      printf("Double precision float)");
      break;
    }
    printf("\nPlease enter a choice: ");

    scanf(" %d", &user_choice);
    switch(user_choice)
    {
    case 0:
      printf("\n\n");
      return;
    case 1:
    case 2:
    case 3:
    case 4:
      // toggle options
      options[user_choice - 1] = !options[user_choice - 1];
      break;
    case 5:
      printf("\n--DATA TYPE--\n"
             "1 - Integer\n"
             "2 - Single precision float\n"
             "3 - Double precision float\n"
             "Please enter a choice: ");
      scanf(" %d", &user_choice);
      switch(user_choice)
      {
        case 1:
          *dt = INT;
          break;
        case 2:
          *dt = FLOAT;
          break;
        case 3:
          *dt = DOUBLE;
          break;
        default:
          printf("Could not recognize your choice of datatype\n");
      }
      set_data_type<<<1,1>>>(*dt);
      break;
    default:
      printf("Could not recognize your choice\n");
    }
    printf("\n");
  }
}

double select_kernel(datatype_t dt, int *selected_kernel_idx)
{
  double expected_result = 0.0;
  for(;;)
  {
    printf("\n--KERNELS--\n"
        "Currently selected: vector %s\n", kernels[*selected_kernel_idx]);
    printf("0 - Back to main menu\n"
           "1 - Vector Addition\n"
           "2 - Vector Subtraction\n"
           "3 - Vector Multiplication\n"
           "4 - Vector Division\n"
           "Please enter a choice: ");

    int user_choice = 0;
    scanf(" %d", &user_choice);

    if(user_choice == 0)
    {
      printf("\n\n");
      return expected_result;
    }
    else if(user_choice > 0 && user_choice < 5)
    {
      // write function pointer to appropriate kernel
      set_kernel_to_run<<<1,1>>>(user_choice - 1);
    }
    else
    {
      printf("Could not recognize your choice\n");
      printf("\n\n");
      return 0.0;
    }

    // make the choice zero-indexed for arrays
    --user_choice;
    // figure out what result to expect based on which kernel will be invoked
    switch(user_choice)
    {
    case 0: //add
      expected_result = 2.0;
      break;
    case 1: //sub
      expected_result = 0.0;
      break;
    case 2: //mult
    case 3: //div
      expected_result = 1.0;
      break;
    default:
      printf("Could not determine kernel specified!");
      continue;
    }
    *selected_kernel_idx = user_choice;
  }
}


int main(int argc, char** argv)
{
  // the number of elements to compute
  const int num_elements = BLK_NUM * BLK_SIZE;

  // the user-selected options
  bool options[OPTION_COUNT];
  // set option defaults
  options[ALL_SM_COMBOS] = true;
  options[COLLECT_LOAD_DIST] = false;
  options[USE_SHARED] = false;
  options[WRITE_REPORT] = true;

  datatype_t selected_data_type = INT;
  int selected_kernel_idx = 0;

  // main menu
  int user_choice = 0;
  double expected_value = 2.0; // addition default, 1+1
  for(;;)
  {
    printf("----GPU Scalability Benchmarks---------------\nSelect an option\n"
            "0 - Quit\n"
            "1 - Edit options\n"
            "2 - Select kernel to run\n"
            "3 - Run benchmark\n"
            "Please enter a choice: ");
    scanf(" %d" , &user_choice);

    switch(user_choice)
    {
      case 0:
        return 0;
      case 1:
        set_options(options, &selected_data_type);
        break;
      case 2:
        expected_value = select_kernel(selected_data_type, &selected_kernel_idx);
        break;
      case 3:
        run_test(num_elements, options, selected_data_type, expected_value);
        break;
    }
  }
}
