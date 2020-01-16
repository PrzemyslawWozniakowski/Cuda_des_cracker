
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>
#include <ctime>
#include <cstdio>
#include "Constants.cuh"
#include "deviceAndHostFunctions.cuh"
#include "deviceFunctions.cuh"
#include "hostFunctions.cuh"

#pragma region CUDA_WRAPPERS

void cudaCheckErrors(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess) {
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
	}
}
#pragma endregion

//__device__ __host__ uint64_t GetNBit(uint64_t number, int bitNumber);
//__device__ __host__ void SetNBit(uint64_t* number, int bitNumber, uint64_t value);
//__device__ __host__ uint64_t ApplyPermutation(uint64_t number, int* Permutation_Table, int length);
//__device__ __host__ void SplitInHalf(uint64_t key, uint64_t* left, uint64_t* right, int keyLength);
//__device__ __host__ uint64_t CycleToLeft(uint64_t value, int shiftNumber, int valueLength);
//
//__host__ uint64_t EncryptData(uint64_t dataToEncrypt, uint64_t desKey);
//__host__  void GenerateSubKeys(uint64_t* subKeys, uint64_t desKey);
//__host__  void GenerateKn(uint64_t* subkeys, uint64_t* C, uint64_t* D);
//__host__ uint64_t Function(uint64_t data, uint64_t key);
//__host__ uint64_t Encode(uint64_t* subKeys, uint64_t dataToEncrypt);
//__host__ void Crack_Host(uint64_t* crackedKey, uint64_t dataToEncrypt, uint64_t encryptedMessage, uint64_t maxKeyVal, int keyLength);
//
//__device__ uint64_t EncryptData_Device(uint64_t dataToEncrypt, uint64_t desKey);
//__device__  void GenerateSubKeys_Device(uint64_t* subKeys, uint64_t desKey);
//__device__  void GenerateKn_Device(uint64_t* subkeys, uint64_t* C, uint64_t* D);
//__device__ uint64_t Function_Device(uint64_t data, uint64_t key);
//__device__ uint64_t Encode_Device(uint64_t* subKeys, uint64_t dataToEncrypt);

__host__ void Crack_Host(uint64_t* crackedKey, uint64_t dataToEncrypt, uint64_t encryptedMessage, uint64_t maxKeyVal, int keyLength);
__host__ void PrintUint(uint64_t v);
__global__ void Crack_Kernel(uint64_t data, uint64_t encodedData, uint64_t *crackedkey, bool *foundFlag, uint64_t maxKeyVal, int keyLength);
__host__ uint64_t GenerateDesKey(int keyLenght);


int main()
{

	cudaCheckErrors(cudaSetDevice(0));
	std::cout << "Dlugosc klucza:" << std::endl;

	int keyLength;
	std::cin >> keyLength;
	uint64_t maxKeyVal = (uint64_t)1 << keyLength;
	uint64_t desKey = GenerateDesKey(keyLength);
	uint64_t dataToEncrypt = 0x0123456789ABCDEF;
	uint64_t encryptedMessage = EncryptData(dataToEncrypt, desKey);
	
	uint64_t* deviceKey = NULL, crackedKeyGPU;
	int cracked_val = 0;
	bool *wasCracked = NULL;
	cudaCheckErrors(cudaMalloc((void**)&deviceKey, sizeof(uint64_t)));
	cudaCheckErrors(cudaMalloc((void**)&wasCracked, sizeof(int)));
	cudaCheckErrors(cudaMemcpy(wasCracked, &cracked_val, sizeof(int), cudaMemcpyHostToDevice));

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	Crack_Kernel << <4096, 1024 >> > (dataToEncrypt, encryptedMessage, deviceKey, wasCracked, maxKeyVal, keyLength);
	cudaCheckErrors(cudaDeviceSynchronize());
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

	auto gpuExecutionTime = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000000.0;

	cudaCheckErrors(cudaMemcpy(&crackedKeyGPU, deviceKey, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	uint64_t encryptedDataWithKeyFromGPU = EncryptData(dataToEncrypt, crackedKeyGPU);
	if (encryptedDataWithKeyFromGPU == encryptedMessage)
	{
		std::cout << "GPU klucz znaleziony w: " << gpuExecutionTime << " sekund" << std::endl;
		std::cout << "Klucz znaleziony na GPU: " << (crackedKeyGPU >> (MAXL - keyLength));
		PrintUint(crackedKeyGPU);
		std::cout << "Oryginalny klucz: " <<(desKey >> (MAXL - keyLength));
		PrintUint(desKey);
	}
	else if (crackedKeyGPU == 0)
	{
		std::cout << "GPU nie znalazlo klucza." << std::endl << std::endl;
	}
	else
	{
		std::cout << "GPU klucz nie dziala." << std::endl;
	}
	std::cout << "================================================= " << std::endl << std::endl;


	start = std::chrono::system_clock::now();
	uint64_t crackedKeyCPU = -1;
	Crack_Host(&crackedKeyCPU, dataToEncrypt, encryptedMessage, maxKeyVal, keyLength);

	end = std::chrono::system_clock::now();

	auto cpuExecutionTime = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000000.0;

	if (crackedKeyCPU != -1)
	{
		std::cout << "CPU klucz znaleziony w: " << cpuExecutionTime << " sekund" << std::endl;
		std::cout << "Klucz znaleziony na CPU: " << (crackedKeyCPU >> (MAXL - keyLength));
		PrintUint(crackedKeyCPU);
		std::cout << "Oryginalny klucz: " << (desKey >> (MAXL - keyLength));
		PrintUint(desKey);
	}
	else
	{
		std::cout << "CPU klucz nie dziala." << std::endl;
	}

	std::cout << "GPU znajduje klucz w " << gpuExecutionTime / cpuExecutionTime * 100 << " % czasu CPU." << std::endl;

	cudaFree(deviceKey);
	cudaFree(wasCracked);

	return 0;
}

__global__ void Crack_Kernel(uint64_t data, uint64_t encodedData, uint64_t *crackedkey, bool *foundFlag, uint64_t maxKeyVal, int keyLength)
{
	for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i <= maxKeyVal; i += blockDim.x * gridDim.x)
	{
		uint64_t keycandidate = i << (MAXL - keyLength);
		uint64_t currentValue = EncryptData_Device(data, keycandidate);
		if (currentValue == encodedData)
		{
			*crackedkey = keycandidate;
			*foundFlag = false;
			return;
		}
		if (*foundFlag == true)
		{
			return;
		}
	}
}

__host__ void Crack_Host(uint64_t* crackedKey, uint64_t dataToEncrypt, uint64_t encryptedMessage, uint64_t maxKeyVal, int keyLength)
{
	for (uint64_t i = 0; i < maxKeyVal; i++)
	{
		uint64_t keycandidate = i << (MAXL - keyLength);
		uint64_t currentValue = EncryptData(dataToEncrypt, keycandidate);
		if (currentValue == encryptedMessage)
		{
			*crackedKey = keycandidate;
			break;
		}
	}
}

__host__ void PrintUint(uint64_t v)
{
	std::cout << "\n";
	uint64_t j = 1;
	for (int i = 0; i < 64; i++)
	{
		std::cout << (v>>(63-i) &j);
		if ((i + 1) % 8 == 0)
			std::cout << " ";
	}
	std::cout << "\n";

}

//#pragma region DeviceAndHostFunctions
//
//
//__device__ __host__ uint64_t GetNBit(uint64_t number, int bitNumber)
//{
//	return ((uint64_t)1 & (number>>(MAXL-bitNumber)));
//}
//
//__device__ __host__ void SetNBit(uint64_t* number, int bitNumber, uint64_t value)
//{
//	(*number) = (*number) &  ~((uint64_t)1 << (MAXL - bitNumber));
//	(*number)= (*number) | (value << (MAXL - bitNumber));
//}
//
//__device__ __host__ uint64_t ApplyPermutation(uint64_t number, int* Permutation_Table, int length)
//{
//	uint64_t numberchanged = 0;
//	for (int i = 0; i < length; i++)
//	{
//		SetNBit(&numberchanged, i+1, GetNBit(number, Permutation_Table[i]));
//	}
//	return numberchanged;	
//}
//
//__device__ __host__ void SplitInHalf(uint64_t key, uint64_t* left, uint64_t* right, int keyLength)
//{
//	*right = *left = 0;
//	for (int i = 1; i <= keyLength / 2; i++)
//	{
//		SetNBit(right, i, GetNBit(key, keyLength / 2 + i));
//		SetNBit(left, i, GetNBit(key, i));
//	}
//}
//
//__device__ __host__ uint64_t CycleToLeft(uint64_t value, int shiftNumber, int valueLength) 
//{
//	for (int i = 0; i < shiftNumber; i++)
//	{
//		uint64_t bit = GetNBit(value, 1);
//		value <<= 1;
//		SetNBit(&value, valueLength, bit);
//	}
//	return value;
//}
//
//#pragma endregion

//#pragma region DeviceFunctions
//
//
//__device__  uint64_t EncryptData_Device(uint64_t dataToEncrypt, uint64_t desKey)
//{
//	uint64_t subKeys[16];
//
//	GenerateSubKeys_Device(subKeys, desKey);
//	uint64_t  encoded = Encode_Device(subKeys, dataToEncrypt);
//	return encoded;
//}
//
//__device__  void GenerateSubKeys_Device(uint64_t* subKeys, uint64_t desKey)
//{
//	uint64_t kplus = ApplyPermutation(desKey, PC_1, 56);
//	uint64_t C[17];
//	uint64_t D[17];
//
//	SplitInHalf(kplus, &C[0], &D[0], 56);
//
//	for (int i = 1; i <= 16; i++)
//	{
//		C[i] = CycleToLeft(C[i - 1], SHIFTS[i - 1], 28);
//		D[i] = CycleToLeft(D[i - 1], SHIFTS[i - 1], 28);
//	}
//
//	GenerateKn_Device(subKeys, C, D);
//}
//
//__device__  void GenerateKn_Device(uint64_t* subkeys, uint64_t* C, uint64_t* D)
//{
//	for (int i = 0; i < 16; i++)
//	{
//		subkeys[i] = C[i + 1];
//		subkeys[i] = subkeys[i] | (D[i + 1] >> 28);
//		subkeys[i] = ApplyPermutation(subkeys[i], PC_2, 48);
//	}
//
//}
//
//__device__  uint64_t Encode_Device(uint64_t* subKeys, uint64_t data)
//{
//	uint64_t data_ip = ApplyPermutation(data, IP, 64);
//
//	uint64_t L[17];
//	uint64_t R[17];
//
//	SplitInHalf(data_ip, &L[0], &R[0], 64);
//
//
//	for (int i = 1; i <= 16; i++)
//	{
//
//		L[i] = R[i - 1];
//		R[i] = L[i - 1] ^ Function_Device(R[i - 1], subKeys[i - 1]);
//	}
//	uint64_t RL = R[16] | (L[16] >> 32);
//	return ApplyPermutation(RL, IP_REV, 64);
//}
//
//__device__  uint64_t Function_Device(uint64_t data, uint64_t key)
//{
//	uint64_t ER = ApplyPermutation(data, E_BIT, 48);
//	uint64_t KxorER = ER ^ key;
//	uint64_t S[8];
//	uint64_t B[8];
//	for (int i = 0; i < 8; i++)
//	{
//		B[i] = 0;
//
//		for (int j = 1; j <= 6; j++)
//		{
//			SetNBit(&B[i], j, GetNBit(KxorER, i * 6 + j));
//		}
//		uint64_t firstLastBit = GetNBit(B[i], 1) << 1 | GetNBit(B[i], 6);
//		uint64_t midBits = GetNBit(B[i], 2) << 3 | GetNBit(B[i], 3) << 2 | GetNBit(B[i], 4) << 1 | GetNBit(B[i], 5);
//		S[i] = ALL_S[i][(int)firstLastBit * 16 + (int)midBits];
//	}
//	uint64_t result = 0;
//
//	for (int i = 0; i < 8; i++)
//	{
//		result |= S[i] << 60 - 4 * i;
//
//	}
//	return ApplyPermutation(result, P, 32);
//}
//
//
//#pragma endregion

//#pragma region HostFunctions
//
//__host__ uint64_t GenerateDesKey(int keyLenght)
//{
//	std::mt19937 mt(time(0));
//	std::uniform_int_distribution<int> randomV(0, 1);
//
//	uint64_t key = 0;
//	for (int i = 1; i <= keyLenght; i++)
//	{
//		SetNBit(&key,i, randomV(mt));
//	}
//	return key;
//}
//
//__host__  uint64_t EncryptData(uint64_t dataToEncrypt, uint64_t desKey)
//{
//	uint64_t subKeys[16];
//
//	GenerateSubKeys(subKeys, desKey);
//	return Encode(subKeys, dataToEncrypt);
//}
//
//__host__  void GenerateSubKeys(uint64_t* subKeys, uint64_t desKey)
//{
//	uint64_t kplus = ApplyPermutation(desKey, PC_1_HOST, 56);
//	uint64_t C[17];
//	uint64_t D[17];
//
//	SplitInHalf(kplus, &C[0], &D[0], 56);
//
//	for (int i = 1; i <= 16; i++)
//	{
//		C[i] = CycleToLeft(C[i - 1], SHIFTS_HOST[i-1], 28);
//		D[i] = CycleToLeft(D[i - 1], SHIFTS_HOST[i-1], 28);
//	}
//	
//	GenerateKn(subKeys,C, D);
//}
//
//__host__  void GenerateKn(uint64_t* subkeys, uint64_t* C, uint64_t* D)
//{
//	
//	for (int i = 0; i < 16; i++)
//	{
//		subkeys[i] = C[i + 1];
//		subkeys[i] = subkeys[i] | (D[i + 1] >> 28);
//		subkeys[i] = ApplyPermutation(subkeys[i], PC_2_HOST, 48);
//	}
//	
//}
//
//__host__  uint64_t Encode(uint64_t* subKeys, uint64_t data)
//{
//	uint64_t data_ip = ApplyPermutation(data, IP_HOST, 64);
//
//	uint64_t L[17];
//	uint64_t R[17];
//
//	SplitInHalf(data_ip, &L[0], &R[0], 64);
//
//
//	for (int i = 1; i <= 16; i++)
//	{
//
//		L[i] = R[i - 1];
//		R[i] = L[i - 1] ^ Function(R[i - 1], subKeys[i - 1]);
//	}
//	uint64_t RL = R[16] | (L[16] >> 32);
//	return ApplyPermutation(RL, IP_REV_HOST, 64);
//}
//
//__host__  uint64_t Function(uint64_t data, uint64_t key)
//{
//	uint64_t ER = ApplyPermutation(data, E_BIT_HOST, 48);
//	uint64_t KxorER = ER ^ key;
//	uint64_t S[8];
//	uint64_t B[8];
//	for (int i = 0; i < 8; i++)
//	{
//		B[i] = 0;
//			
//		for (int j = 1; j <= 6; j++)
//		{
//			SetNBit(&B[i], j, GetNBit(KxorER, i * 6 + j));
//		}
//		uint64_t firstLastBit = GetNBit(B[i], 1) << 1 | GetNBit(B[i], 6);
//		uint64_t midBits = GetNBit(B[i],2) << 3 | GetNBit(B[i], 3) << 2 | GetNBit(B[i], 4) << 1 | GetNBit(B[i],5);
//		S[i] = ALL_S_HOST[i][(int)firstLastBit * 16 + (int)midBits];
//	}
//	uint64_t result = 0;
//
//	for (int i = 0; i < 8; i++)
//	{
//		result |= S[i] << 60 - 4 * i;
//
//	}	
//	return ApplyPermutation(result, P_HOST, 32);
//}

#pragma endregion