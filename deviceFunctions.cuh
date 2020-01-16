#ifndef _DEVICEFUNCTIONS_CUH_
#define _DEVICEFUNCTIONS_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include "Constants.cuh"
#include "deviceAndHostFunctions.cuh"


__device__ uint64_t EncryptData_Device(uint64_t dataToEncrypt, uint64_t desKey);
__device__  void GenerateSubKeys_Device(uint64_t* subKeys, uint64_t desKey);
__device__  void GenerateKn_Device(uint64_t* subkeys, uint64_t* C, uint64_t* D);
__device__ uint64_t Function_Device(uint64_t data, uint64_t key);
__device__ uint64_t Encode_Device(uint64_t* subKeys, uint64_t dataToEncrypt);

#pragma region DeviceFunctions
__device__  uint64_t EncryptData_Device(uint64_t dataToEncrypt, uint64_t desKey)
{
	uint64_t subKeys[16];

	GenerateSubKeys_Device(subKeys, desKey);
	uint64_t  encoded = Encode_Device(subKeys, dataToEncrypt);
	return encoded;
}

__device__  void GenerateSubKeys_Device(uint64_t* subKeys, uint64_t desKey)
{
	uint64_t kplus = ApplyPermutation(desKey, PC_1, 56);
	uint64_t C[17];
	uint64_t D[17];

	SplitInHalf(kplus, &C[0], &D[0], 56);

	for (int i = 1; i <= 16; i++)
	{
		C[i] = CycleToLeft(C[i - 1], SHIFTS[i - 1], 28);
		D[i] = CycleToLeft(D[i - 1], SHIFTS[i - 1], 28);
	}

	GenerateKn_Device(subKeys, C, D);
}

__device__  void GenerateKn_Device(uint64_t* subkeys, uint64_t* C, uint64_t* D)
{
	for (int i = 0; i < 16; i++)
	{
		subkeys[i] = C[i + 1];
		subkeys[i] = subkeys[i] | (D[i + 1] >> 28);
		subkeys[i] = ApplyPermutation(subkeys[i], PC_2, 48);
	}

}

__device__  uint64_t Encode_Device(uint64_t* subKeys, uint64_t data)
{
	uint64_t data_ip = ApplyPermutation(data, IP, 64);

	uint64_t L[17];
	uint64_t R[17];

	SplitInHalf(data_ip, &L[0], &R[0], 64);


	for (int i = 1; i <= 16; i++)
	{

		L[i] = R[i - 1];
		R[i] = L[i - 1] ^ Function_Device(R[i - 1], subKeys[i - 1]);
	}
	uint64_t RL = R[16] | (L[16] >> 32);
	return ApplyPermutation(RL, IP_REV, 64);
}

__device__  uint64_t Function_Device(uint64_t data, uint64_t key)
{
	uint64_t ER = ApplyPermutation(data, E_BIT, 48);
	uint64_t KxorER = ER ^ key;
	uint64_t S[8];
	uint64_t B[8];
	for (int i = 0; i < 8; i++)
	{
		B[i] = 0;

		for (int j = 1; j <= 6; j++)
		{
			SetNBit(&B[i], j, GetNBit(KxorER, i * 6 + j));
		}
		uint64_t firstLastBit = GetNBit(B[i], 1) << 1 | GetNBit(B[i], 6);
		uint64_t midBits = GetNBit(B[i], 2) << 3 | GetNBit(B[i], 3) << 2 | GetNBit(B[i], 4) << 1 | GetNBit(B[i], 5);
		S[i] = ALL_S[i][(int)firstLastBit * 16 + (int)midBits];
	}
	uint64_t result = 0;

	for (int i = 0; i < 8; i++)
	{
		result |= S[i] << 60 - 4 * i;

	}
	return ApplyPermutation(result, P, 32);
}


#pragma endregion
#endif