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

int estimateNumElements(int SizeA, int SizeB, int SizeC, int SizeD, int SizeTmp, double AllowedSpaceInGB);

int main(int Argc, char* Arcv[]) {

  YAML::Node Params = YAML::LoadFile("./params.yaml");
  YAML::Node MatrixASpec = Params["MatA"];
  YAML::Node MatrixBSpec = Params["MatB"];
  YAML::Node MatrixCSpec = Params["MatC"];
  YAML::Node MatrixDSpec = Params["MatD"];

  int SizeA = MatrixASpec["num_rows"].as<int>() * MatrixASpec["num_cols"].as<int>();
  int SizeB = MatrixBSpec["num_rows"].as<int>() * MatrixBSpec["num_cols"].as<int>();
  int SizeC = MatrixCSpec["num_rows"].as<int>() * MatrixCSpec["num_cols"].as<int>();
  int SizeD = MatrixDSpec["num_rows"].as<int>() * MatrixDSpec["num_cols"].as<int>();

  std::vector<int> BboxA = MatrixASpec["bbox"].as<std::vector<int>>();
  std::vector<int> BboxB = MatrixBSpec["bbox"].as<std::vector<int>>();
  std::vector<int> BboxC = MatrixCSpec["bbox"].as<std::vector<int>>();
  std::vector<int> BboxD = MatrixCSpec["bbox"].as<std::vector<int>>();

  int L = BboxC[2] - BboxC[0];
  int M = BboxA[2] - BboxA[0];
  int N = BboxB[3] - BboxB[1];
  int K = BboxA[3] - BboxA[1];

  int SizeTemp = M * N;  // !< required only the exact size

  real Alpha = Params["alpha"].as<real>();
  real Beta = Params["beta"].as<real>();

  YAML::Node Config = YAML::LoadFile("./config.yaml");
  int NumElements = estimateNumElements(SizeA, SizeB, SizeC, SizeD, SizeTemp, Config["allocate_mem"].as<double>());

  long long FlopCounter = computeNumFlops(M, N, K, 1.0, 0.0);
  FlopCounter += computeNumFlops(L, N, M, Alpha, Beta);

  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(0);
  device.api->initialize();

  dense::TestDriver FirstDriver(SizeA, SizeB, SizeTemp, NumElements);
  dense::TestDriver SecondDriver(SizeC, SizeTemp, SizeTemp, NumElements);
  auto TotalMem = FirstDriver.getDeviceAllocatedMemSize() + SecondDriver.getDeviceAllocatedMemSize();

  std::cout << "Allocated Device Mem. GB: " << TotalMem << std::endl;

  FirstDriver.SetUp();
  SecondDriver.SetUp();

  real *HostA{};
  real *HostB{};
  real *HostTmp{};
  std::tie(HostA, HostB, HostTmp) = FirstDriver.getHostRawData();

  real *DeviceA{};
  real *DeviceB{};
  real *DeviceTmp{};
  std::tie(DeviceA, DeviceB, DeviceTmp) = FirstDriver.getDeviceRawData();

  real *HostC{};
  real *HostD{};
  std::tie(HostC, std::ignore, HostD) = FirstDriver.getHostRawData();

  real *DeviceC{};
  real *DeviceD{};
  std::tie(DeviceC, std::ignore, DeviceD) = FirstDriver.getDeviceRawData();


  // Check correctness
  std::cout << "INFO: computing on CPU started" << std::endl;
  unsigned NextTmp = SizeTemp;
  unsigned NextD = SizeD;

  unsigned NextA = MatrixASpec["addressing"].as<std::string>() == std::string("strided") ? SizeA : 0;
  unsigned NextB = MatrixBSpec["addressing"].as<std::string>() == std::string("strided") ? SizeB : 0;
  unsigned NextC = MatrixCSpec["addressing"].as<std::string>() == std::string("strided") ? SizeC : 0;

  LayoutType TransA = Params["trans_a"].as<bool>() ? LayoutType::Trans : LayoutType::NoTrans;
  LayoutType TransB = Params["trans_b"].as<bool>() ? LayoutType::Trans : LayoutType::NoTrans;
  LayoutType TransC = Params["trans_c"].as<bool>() ? LayoutType::Trans : LayoutType::NoTrans;

  int Lda = MatrixASpec["num_rows"].as<int>();
  int Ldb = MatrixBSpec["num_rows"].as<int>();
  int Ldc = MatrixCSpec["num_rows"].as<int>();
  int Ldd = MatrixDSpec["num_rows"].as<int>();
  int LdTemp = M;

  auto computeOffset = [](const int LidDim, const std::vector<int> &Bbox) {
    return LidDim * Bbox[1] + Bbox[0];
  };

  int OffsetA = computeOffset(Lda, BboxA);
  int OffsetB = computeOffset(Ldb, BboxB);
  int OffsetC = computeOffset(Ldc, BboxC);
  int OffsetD = computeOffset(Ldd, BboxD);
  int OffsetTemp = 0;


  gemmforge::reference::gemm(TransA, TransB,
                             M, N, K,
                             1.0, &HostA[OffsetA], Lda,
                             &HostB[OffsetB], Ldb,
                             0.0, HostTmp, LdTemp,
                             NextA, NextB, NextTmp,
                             NumElements);


  gemmforge::reference::gemm(TransC, reference::LayoutType::NoTrans,
                             L, N, M,
                             Alpha, &HostC[OffsetC], Ldc,
                             HostTmp, M,
                             Beta, &HostD[OffsetD], Ldd,
                             NextC, NextTmp, NextD,
                             NumElements);

  std::cout << "INFO: computing on GPU started" << std::endl;
  callFirstGemm(DeviceA, 0, DeviceB, 0, DeviceTmp, 0, NumElements, nullptr, FirstDriver.getTestStream());
  callSecondGemm(DeviceC, 0, DeviceTmp, 0, DeviceD, 0, NumElements, nullptr, SecondDriver.getTestStream());
  synchDevice(FirstDriver.getTestStream());
  synchDevice(SecondDriver.getTestStream());

  std::cout << "INFO: comparsion started" << std::endl;

  SecondDriver.packResults(L,  Ldd, N, OffsetD, SizeD, NumElements);
  bool IsPassed = SecondDriver.isTestPassed<SimpleComparator>();
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
    callFirstGemm(DeviceA, 0, DeviceB, 0, DeviceTmp, 0, NumElements, nullptr, FirstDriver.getTestStream());
    callSecondGemm(DeviceC, 0, DeviceTmp, 0, DeviceD, 0, NumElements, nullptr, SecondDriver.getTestStream());
  }
  synchDevice(FirstDriver.getTestStream());
  synchDevice(SecondDriver.getTestStream());

  Timer.stop();

  std::cout << "Num elements: " << NumElements << std::endl;
  std::cout << "Num repeats: " << NumRepeats << std::endl;
  std::cout << "Computed Flops: " << NumRepeats * FlopCounter << std::endl;
  std::cout << "Spent time: " << Timer.getTime() << std::endl;
  std::cout << "GFLOPS: " << NumRepeats * FlopCounter / (Timer.getTime() / NumElements) << std::endl;

  FirstDriver.TearDown();
  SecondDriver.TearDown();
  device.api->finalize();
  return 0;

}


int estimateNumElements(int SizeA, int SizeB, int SizeC, int SizeD, int SizeTmp, double AllowedSpaceInGB) {
  // Note: We are going to use only one matrix C. However, memory is going
  // to get allocated for all elements
  long long ElementSizeInBytes = (SizeD + SizeC + SizeTmp + SizeA + SizeB) * sizeof(real);
  constexpr double FACTOR = 1024 * 1024 * 1024;
  return int((AllowedSpaceInGB * FACTOR) / ElementSizeInBytes);
}