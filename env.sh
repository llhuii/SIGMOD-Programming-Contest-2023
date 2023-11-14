#!/bin/bash

base=/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/
base=/opt/intel/oneapi/mkl/latest
base=compilers_and_libraries_2020.4.304 
base=/opt/intel/$base/linux/mkl/
base=/opt/intel/mkl

#LD_LIBRARY_PATH=

export LD_LIBRARY_PATH=$base/lib/intel64:$PWD/nn-descent:.:/usr/local/lib
export C_INCLUDE_PATH=$base/include:$C_INCLUDE_PATH
export MKLROOT=$base
export MKL_INCLUDE_DIR=$base/include
export MKL_LIBRARY_DIRS=$base/lib/intel64

export CMAKE_INCLUDE_PATH=$C_INCLUDE_PATH



function run() {
#make clean
#make
./knng
}

