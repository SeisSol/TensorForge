#include "hip/hip_runtime.h"
#include "stop_watch.h"
#include "gemmgen_aux.h"
#include "stop_watch.h"
#include "yaml-cpp/yaml.h"
#include <iostream>

using namespace gemmgen;

__global__ void copyData(float *To, float *From, size_t NumElements) {
  int Idx = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
  if (Idx < NumElements) {
    To[Idx] = From[Idx];
  }
}


int main(int Argc, char *Argv[]) {
  YAML::Node Config = YAML::LoadFile("../config.yaml");
  int NumRepeats = Config["num_repeats"].as<int>();
  double AllocatedMemGb = Config["allocated_mem"].as<double>();

  constexpr long long FACTOR = 1024 * 1024 * 1024;
  size_t NumElements = (FACTOR * AllocatedMemGb) / sizeof(float);

  float *To = nullptr;
  float *From = nullptr;

  hipMalloc(&To, NumElements * sizeof(float)); CHECK_ERR;
  hipMalloc(&From, NumElements * sizeof(float)); CHECK_ERR;

  dim3 Block(1024, 1, 1);
  dim3 Grid((NumElements + 1024 - 1) / 1024, 1, 1);

  utils::StopWatch<std::chrono::duration<double, std::chrono::nanoseconds::period>> Timer;
  Timer.start();
  for (int Repeat = 0; Repeat < NumRepeats; ++Repeat) {
    hipLaunchKernelGGL(copyData, dim3(Grid), dim3(Block), 0, 0, To, From, NumElements);
  }
  synchDevice();

  Timer.stop();
  CHECK_ERR;

  auto AverageTime = Timer.getTime() / NumRepeats;
  // 1 copy and 1 write explains the factor of 2
  double BandwidthGb = 2 * (NumElements / AverageTime) * sizeof(float);
  std::cout << "Allocated Mem, GB: " << AllocatedMemGb << std::endl;
  std::cout << "Time: " << Timer.getTime() << std::endl;
  std::cout << "Num. Repeats: " << NumRepeats << std::endl;
  std::cout << "Num. Elements: " << NumElements << std::endl;
  std::cout << "Achieved bandwidth: " << BandwidthGb << " GB/s" << std::endl;

  hipFree(To); CHECK_ERR;
  hipFree(From); CHECK_ERR;

  return 0;
}
