
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <ctime>

#include "Constants.cuh"
#include "deviceAndHostFunctions.cuh"

__host__ uint64_t EncryptData(uint64_t dataToEncrypt, uint64_t desKey);
__host__  void GenerateSubKeys(uint64_t* subKeys, uint64_t desKey);
__host__  void GenerateKn(uint64_t* subkeys, uint64_t* C, uint64_t* D);
__host__ uint64_t Function(uint64_t data, uint64_t key);
__host__ uint64_t Encode(uint64_t* subKeys, uint64_t dataToEncrypt);

#pragma region HostFunctions

__host__ uint64_t GenerateDesKey(int keyLenght)
{
	std::mt19937 mt(time(0));
	std::uniform_int_distribution<int> randomV(0, 1);

	uint64_t key = 0;
	for (int i = 1; i <= keyLenght; i++)
	{
		SetNBit(&key, i, randomV(mt));
	}
	return key;
}

__host__  uint64_t EncryptData(uint64_t dataToEncrypt, uint64_t desKey)
{
	uint64_t subKeys[16];

	GenerateSubKeys(subKeys, desKey);
	return Encode(subKeys, dataToEncrypt);
}

__host__  void GenerateSubKeys(uint64_t* subKeys, uint64_t desKey)
{
	uint64_t kplus = ApplyPermutation(desKey, PC_1_HOST, 56);
	uint64_t C[17];
	uint64_t D[17];

	SplitInHalf(kplus, &C[0], &D[0], 56);

	for (int i = 1; i <= 16; i++)
	{
		C[i] = CycleToLeft(C[i - 1], SHIFTS_HOST[i - 1], 28);
		D[i] = CycleToLeft(D[i - 1], SHIFTS_HOST[i - 1], 28);
	}

	GenerateKn(subKeys, C, D);
}

__host__  void GenerateKn(uint64_t* subkeys, uint64_t* C, uint64_t* D)
{

	for (int i = 0; i < 16; i++)
	{
		subkeys[i] = C[i + 1];
		subkeys[i] = subkeys[i] | (D[i + 1] >> 28);
		subkeys[i] = ApplyPermutation(subkeys[i], PC_2_HOST, 48);
	}

}

__host__  uint64_t Encode(uint64_t* subKeys, uint64_t data)
{
	uint64_t data_ip = ApplyPermutation(data, IP_HOST, 64);

	uint64_t L[17];
	uint64_t R[17];

	SplitInHalf(data_ip, &L[0], &R[0], 64);


	for (int i = 1; i <= 16; i++)
	{

		L[i] = R[i - 1];
		R[i] = L[i - 1] ^ Function(R[i - 1], subKeys[i - 1]);
	}
	uint64_t RL = R[16] | (L[16] >> 32);
	return ApplyPermutation(RL, IP_REV_HOST, 64);
}

__host__  uint64_t Function(uint64_t data, uint64_t key)
{
	uint64_t ER = ApplyPermutation(data, E_BIT_HOST, 48);
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
		S[i] = ALL_S_HOST[i][(int)firstLastBit * 16 + (int)midBits];
	}
	uint64_t result = 0;

	for (int i = 0; i < 8; i++)
	{
		result |= S[i] << 60 - 4 * i;

	}
	return ApplyPermutation(result, P_HOST, 32);
}

#pragma endregion