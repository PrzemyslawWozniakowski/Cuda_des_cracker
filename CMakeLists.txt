# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)

project(desCracker LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)
 
# Specify binary name and source file to build it from
cuda_add_executable(
	descracker
	descracker.cu
	Constants.cuh
	deviceAndHostFunctions.cuh
	deviceFunctions.cuh
	hostFunctions.cuh
	)