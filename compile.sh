#!/bin/bash

# -03
if [ `hostname` == "eecs-hpc-1" ]
then
  echo "Compiling for eecs-hpc-1 (Teslsa)"
  nvcc -g -G -D NUM_SMS=15 -o gpuscale gpuscale.cu -arch=sm_35
else
  echo "Compiling for Issac (GTX 660)"
  nvcc -g -G -D NUM_SMS=5 -o gpuscale gpuscale.cu -arch=sm_30
fi
