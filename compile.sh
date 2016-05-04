#!/bin/bash

# Makefile-esqe defines
cc='nvcc'
#flags='-g -G'
flags='-O3'
app='-o gpuscale gpuscale.cu'

if [ `hostname` == "eecs-hpc-1" ]
then
  echo "Compiling for eecs-hpc-1 (Teslsa)"
  $cc $flags $app -D NUM_SMS=15 -arch=sm_35
else
  echo "Compiling for Issac (GTX 660)"
  $cc $flags $app -D NUM_SMS=5 -arch=sm_30
fi
