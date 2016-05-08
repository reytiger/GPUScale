#!/bin/bash

# Makefile-esqe defines
cc='nvcc'
#flags='-g -G -x cu'
flags='-O3 -x cu'
app='-o gpuscale gpuscale.c kernels.cu'

if [ `hostname` == "eecs-hpc-1" ]
then
  echo "Compiling for eecs-hpc-1 (Tesla)"
  plat='-D NUM_SMS=15 -arch=sm_35'
else
  echo "Compiling for Issac (GTX 660)"
  plat='-D NUM_SMS=5 -arch=sm_30'
fi
$cc $flags $plat $app
