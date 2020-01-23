
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

	uint64_t maxKeyVal;

	if (keyLength == 64)
		maxKeyVal = UINT64_MAX;
	else
	{
		maxKeyVal = (uint64_t)1 << keyLength;
		maxKeyVal -= 1;
	}
	uint64_t desKey = GenerateDesKey(keyLength);
	uint64_t dataToEncrypt = 0x0123456789ABCDEF;
	uint64_t encryptedMessage = EncryptData(dataToEncrypt, desKey);

	std::cout << "Wiadomosc: ";
	PrintUint(dataToEncrypt);
	std::cout << "Wiadomosc po zaszyfrowaniu: ";
	PrintUint(encryptedMessage);

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
		std::cout << "Oryginalny klucz: " << (desKey >> (MAXL - keyLength));
		PrintUint(desKey);
		std::cout << "Wiadomosc po zaszyfrowaniu kluczem z GPU: ";
		PrintUint(encryptedDataWithKeyFromGPU);
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
	for (uint64_t i = 0; i <= maxKeyVal; i++)
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
		std::cout << (v >> (63 - i) &j);
		if ((i + 1) % 8 == 0)
			std::cout << " ";
	}
	std::cout << "\n";
}


#pragma endregion