#include "aux.h"
#include "comparators.h"
#include "simple_driver.h"
#include "kernels.h"
#include "stop_watch.h"
#include "gemm.h"
#include "gemmforge_aux.h"
#include "yaml-cpp/yaml.h"
#include <device.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>

using namespace gemmforge;
using namespace reference;
using namespace device;

int estimateNumElements(int SizeA, int SizeB, int SizeC, double AllowedSpaceInGB);

int main(int Argc, char* Arcv[]) {


  YAML::Node Params = YAML::LoadFile("./params.yaml");
  YAML::Node MatrixASpec = Params["MatA"];
  YAML::Node MatrixBSpec = Params["MatB"];
  YAML::Node MatrixCSpec = Params["MatC"];

  int SizeA = MatrixASpec["num_rows"].as<int>() * MatrixASpec["num_cols"].as<int>();
  int SizeB = MatrixBSpec["num_rows"].as<int>() * MatrixBSpec["num_cols"].as<int>();
  int SizeC = MatrixCSpec["num_rows"].as<int>() * MatrixCSpec["num_cols"].as<int>();

  std::vector<int> BboxA = MatrixASpec["bbox"].as<std::vector<int>>();
  std::vector<int> BboxB = MatrixBSpec["bbox"].as<std::vector<int>>();
  std::vector<int> BboxC = MatrixCSpec["bbox"].as<std::vector<int>>();

  int M = BboxA[2] - BboxA[0];
  int N = BboxB[3] - BboxB[1];
  int K = BboxA[3] - BboxA[1];

  real Alpha = Params["alpha"].as<real>();
  real Beta = Params["beta"].as<real>();

  YAML::Node Config = YAML::LoadFile("./config.yaml");
  int NumElements = estimateNumElements(SizeA, SizeB, SizeC, Config["allocate_mem"].as<double>());

  long long FlopCounter = computeNumFlops(M, N, K, Alpha, Beta);

  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(0);
  device.api->initialize();

  dense::TestDriver Driver(SizeA, SizeB, SizeC, NumElements);
  std::cout << "Allocated Device Mem. GB: " << Driver.getDeviceAllocatedMemSize() << std::endl;
  Driver.SetUp();

  real *HostA{};
  real *HostB{};
  real *HostC{};
  std::tie(HostA, HostB, HostC) = Driver.getHostRawData();

  real *DeviceA{};
  real *DeviceB{};
  real *DeviceC{};
  std::tie(DeviceA, DeviceB, DeviceC) = Driver.getDeviceRawData();

  std::vector<real*> ShuffledDeviceA{};
  std::vector<real*> ShuffledDeviceB{};
  std::vector<real*> ShuffledDeviceC{};
  std::tie(ShuffledDeviceA, ShuffledDeviceB, ShuffledDeviceC) = Driver.getShuffledDeviceData();

  // Check correctness
  std::cout << "INFO: computing on CPU started" << std::endl;
  unsigned NextA = MatrixASpec["addressing"].as<std::string>() == std::string("strided") ? SizeA : 0;
  unsigned NextB = MatrixBSpec["addressing"].as<std::string>() == std::string("strided") ? SizeB : 0;
  unsigned NextC = SizeC;

  LayoutType TransA = Params["trans_a"].as<bool>() ? LayoutType::Trans : LayoutType::NoTrans;
  LayoutType TransB = Params["trans_b"].as<bool>() ? LayoutType::Trans : LayoutType::NoTrans;

  int Lda = MatrixASpec["num_rows"].as<int>();
  int Ldb = MatrixBSpec["num_rows"].as<int>();
  int Ldc = MatrixCSpec["num_rows"].as<int>();

  auto computeOffset = [](const int LidDim, const std::vector<int> &Bbox) {
    return LidDim * Bbox[1] + Bbox[0];
  };

  int OffsetA = computeOffset(Lda, BboxA);
  int OffsetB = computeOffset(Ldb, BboxB);
  int OffsetC = computeOffset(Ldc, BboxC);

  gemmforge::reference::gemm(TransA, TransB,
                             M, N, K,
                             Alpha, &HostA[OffsetA], Lda,
                             &HostB[OffsetB], Ldb,
                             Beta, &HostC[OffsetC], Ldc,
                             NextA, NextB, NextC,
                             NumElements);

  std::cout << "INFO: computing on GPU started" << std::endl;
  gemm(DeviceA, 0, DeviceB, 0, DeviceC, 0, NumElements, nullptr, Driver.getTestStream());
  synchDevice(Driver.getTestStream());

  std::cout << "INFO: comparsion started" << std::endl;
  Driver.packResults(M,  Ldc, N, OffsetC, SizeC, NumElements);
  bool IsPassed = Driver.isTestPassed<SimpleComparator>();
  if (IsPassed) {
    std::cout << "INFO: Results are correct" << std::endl;
  }
  else {
    std::cout << "WARNING: Test failed" << std::endl;
  }

  // Measure performance
  utils::StopWatch<std::chrono::duration<double, std::chrono::nanoseconds::period>> Timer;
  int NumRepeats = Config["num_repeats"].as<int>();
  Timer.start();
  for (int Repeat = 0; Repeat < NumRepeats; ++Repeat) {
    gemm(DeviceA, 0, DeviceB, 0, DeviceC, 0, NumElements, nullptr, Driver.getTestStream());
  }
  synchDevice(Driver.getTestStream());
  Timer.stop();

  std::cout << "Num elements: " << NumElements << std::endl;
  std::cout << "Num repeats: " << NumRepeats << std::endl;
  std::cout << "Computed Flops: " << NumRepeats * FlopCounter * NumElements << std::endl;
  std::cout << "Spent time: " << Timer.getTime() << std::endl;
  std::cout << "GFLOPS: " << NumRepeats * FlopCounter / (Timer.getTime() / NumElements) << std::endl;

  Driver.TearDown();
  device.api->finalize();
  return 0;
}


int estimateNumElements(int SizeA, int SizeB, int SizeC, double AllowedSpaceInGB) {
  long long ElementSizeInBytes = (SizeA + SizeB + SizeC) * sizeof(real);
  constexpr double FACTOR = 1024 * 1024 * 1024;
  return int((AllowedSpaceInGB * FACTOR) / ElementSizeInBytes);
}