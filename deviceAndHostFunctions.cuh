#ifndef _DEVICEANDHOSTFUNCTIONS_CUH_
#define _DEVICEANDHOSTFUNCTIONS_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include "Constants.cuh"

__device__ __host__ uint64_t GetNBit(uint64_t number, int bitNumber);
__device__ __host__ void SetNBit(uint64_t* number, int bitNumber, uint64_t value);
__device__ __host__ uint64_t ApplyPermutation(uint64_t number, int* Permutation_Table, int length);
__device__ __host__ void SplitInHalf(uint64_t key, uint64_t* left, uint64_t* right, int keyLength);
__device__ __host__ uint64_t CycleToLeft(uint64_t value, int shiftNumber, int valueLength);

#pragma region DeviceAndHostFunctions

__device__ __host__ uint64_t GetNBit(uint64_t number, int bitNumber)
{
	return ((uint64_t)1 & (number >> (MAXL - bitNumber)));
}

__device__ __host__ void SetNBit(uint64_t* number, int bitNumber, uint64_t value)
{
	(*number) = (*number) &  ~((uint64_t)1 << (MAXL - bitNumber));
	(*number) = (*number) | (value << (MAXL - bitNumber));
}

__device__ __host__ uint64_t ApplyPermutation(uint64_t number, int* Permutation_Table, int length)
{
	uint64_t numberchanged = 0;
	for (int i = 0; i < length; i++)
	{
		SetNBit(&numberchanged, i + 1, GetNBit(number, Permutation_Table[i]));
	}
	return numberchanged;
}

__device__ __host__ void SplitInHalf(uint64_t key, uint64_t* left, uint64_t* right, int keyLength)
{
	*right = *left = 0;
	for (int i = 1; i <= keyLength / 2; i++)
	{
		SetNBit(right, i, GetNBit(key, keyLength / 2 + i));
		SetNBit(left, i, GetNBit(key, i));
	}
}

__device__ __host__ uint64_t CycleToLeft(uint64_t value, int shiftNumber, int valueLength)
{
	for (int i = 0; i < shiftNumber; i++)
	{
		uint64_t bit = GetNBit(value, 1);
		value <<= 1;
		SetNBit(&value, valueLength, bit);
	}
	return value;
}

#pragma endregion

#endif