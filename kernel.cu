
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
//#include "Const_Values.h"

typedef unsigned long long uint64;
#define MAXL 64
#pragma region CUDA_CONSTANTS

__constant__ int PC_1[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

__constant__ int PC_2[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

__constant__ int IP[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

__constant__ int E_BIT[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

__constant__ int S1[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

__constant__ int S2[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

__constant__ int S3[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

__constant__ int S4[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

__constant__ int S5[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

__constant__ int S6[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

__constant__ int S7[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

__constant__ int S8[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

__constant__ int* ALL_S[8] = {
	S1, S2, S3, S4, S5, S6, S7, S8
};

__constant__ int P[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

__constant__ int IP_REV[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

__constant__ int SHIFTS[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};

#pragma endregion

#pragma region HOST_CONSTANTS

int PC_1_HOST[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

int PC_2_HOST[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

int IP_HOST[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

int E_BIT_HOST[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

int S1_HOST[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

int S2_HOST[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

int S3_HOST[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

int S4_HOST[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

int S5_HOST[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

int S6_HOST[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

int S7_HOST[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

int S8_HOST[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

int* ALL_S_HOST[8] = {
	S1_HOST, S2_HOST, S3_HOST, S4_HOST, S5_HOST, S6_HOST, S7_HOST, S8_HOST
};

int P_HOST[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

int IP_REV_HOST[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

int SHIFTS_HOST[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};
#pragma endregion

void cudaCheckErrors(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess) {
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
	}
}

__device__ __host__ uint64 Get_Bit(uint64 number, int bitNumber);
__device__ __host__ void Set_Bit(uint64* number, int bitNumber, uint64 value);
__device__ __host__ uint64 Permute(uint64 number, int* Permutation_Table, int length);
__device__ __host__ void Split(uint64 key, uint64* left, uint64* right, int keyLength);
__device__ __host__ uint64 CycleBitsToLeft(uint64 value, int shiftNumber, int valueLength);

__device__  void Create_SubKeys(uint64* subKeys, uint64 desKey);
__device__  uint64* Create_Kn(uint64* C, uint64* D);

__device__ uint64 Function(uint64 data, uint64 key);
__device__ uint64 Encode(uint64* subKeys, uint64 dataToEncrypt);

__host__ void PrintUint(uint64 v);

int main()
{
	uint64 osiem = 8, jeden=3;
	uint64 bit = Get_Bit(osiem, 61);
	Set_Bit(&osiem, 64, (uint64)1);
	std::cout << osiem;
	PrintUint(jeden);
	uint64 innaliczba = CycleBitsToLeft(jeden, 2, 63);
	PrintUint(innaliczba);
	return 0;
}

#pragma region DeviceAndHostFunctions

__host__ void PrintUint(uint64 v)
{
	std::cout << "\n";
	uint64 j = 1;
	for (int i = 0; i < 64; i++)
	{
		std::cout << (v>>(63-i) &j);
	}
	std::cout << "\n";

}

__device__ __host__ uint64 Get_Bit(uint64 number, int bitNumber)
{
	return ((uint64)1 & (number>>(MAXL-bitNumber)));
}

__device__ __host__ void Set_Bit(uint64* number, int bitNumber, uint64 value)
{
	(*number) = (*number) &  ~((uint64)1 << (MAXL - bitNumber));
	(*number)= (*number) | (value << (MAXL - bitNumber));
}

__device__ __host__ uint64 Permute(uint64 number, int* Permutation_Table, int length)
{
	uint64 numberchanged = 0;
	for (int i = 0; i < length; i++)
	{
		Set_Bit(&numberchanged, i+1, Get_Bit(number, Permutation_Table[i]));
		PrintUint(numberchanged);
	}
	return numberchanged;	
}

__device__ __host__ void Split(uint64 key, uint64* left, uint64* right, int keyLength)
{
	*right = *left = 0;
	for (int i = 0; i < keyLength / 2; i++)
	{
		Set_Bit(left, i, Get_Bit(key, keyLength / 2 + i));
		Set_Bit(right, i, Get_Bit(key, i));
	}
}

__device__ __host__ uint64 CycleBitsToLeft(uint64 value, int shiftNumber, int valueLength) 
{
	for (int i = 0; i < shiftNumber; i++)
	{
		uint64 bit = Get_Bit(value, 1);
		value <<= 1;
		Set_Bit(&value, valueLength, bit);
	}
	return value;
}

#pragma endregion

#pragma region DeviceFunctions

__device__  void Create_SubKeys(uint64* subKeys, uint64 desKey)
{
	uint64 kplus = Permute(desKey, PC_1, 56);

	uint64 C[17];
	uint64 D[17];

	Split(kplus, &C[0], &D[0], 56);

	for (int i = 1; i <= 16; i++)
	{
		C[i] = CycleBitsToLeft(C[i - 1], SHIFTS[i], 28);
		D[i] = CycleBitsToLeft(C[i - 1], SHIFTS[i], 28);
	}

	subKeys= Create_Kn(C, D);
}

__device__  uint64* Create_Kn(uint64* C, uint64* D)
{
	uint64	Kn[16];
	for(int i=0;i<16;i++)
	{
		Kn[i] = C[i + 1] << 28;
		Kn[i] = Kn[i] | D[i + 1];
		Kn[i] = Permute(Kn[i], PC_2, 48);
	}
}

//__device__ uint64 Encode(uint64* subKeys,uint64 data)
//{
//	uint64 data_ip = Permute(data, IP, 64);
//
//	uint64 L[17];
//	uint64 R[17];
//
//	Split(data_ip, &L[0], &R[0], 64);
//
//	for (int i = 1; i <= 16; i++)
//	{
//		L[i] = R[i-1];
//		R
//	}
//
//}
//
//__device__ uint64 Function(uint64 data, uint64 key)
//{
//	uint64 ER = Permute(data, E_BIT, 48);
//	uint64 KxorER = ER ^ key;
//	uint64 S[8];
//	uint64 B[8];
//
//	for (int i = 0; i < 8; i++)
//	{
//		B[i] = 0;
//
//		for(int j=0; j<)
//	}
//}

#pragma endregion

