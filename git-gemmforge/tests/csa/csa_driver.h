#ifndef GEMMS_GEMM_DRIVER_H
#define GEMMS_GEMM_DRIVER_H

#include "typedef.h"
#include "simple_driver.h"
#include "gtest/gtest.h"
#include <tuple>
#include <vector>

class DenseCsaTest : public ::testing::Test {
protected:
  void SetUp(int M, int N, int K, int NumElements) {
    Driver.setParams(M, N, K, NumElements);
    Driver.SetUp();

    std::tie(HostA, HostB, HostC) = Driver.getHostRawData();
    std::tie(DeviceA, DeviceB, DeviceC) = Driver.getDeviceRawData();
    std::tie(DeviceShuffledA, DeviceShuffledB, DeviceShuffledC) = Driver.getShuffledDeviceData();
  }

  void TearDown() {
    Driver.TearDown();
  }

  real *HostA = nullptr;
  real *HostB = nullptr;
  real *HostC = nullptr;

  real *DeviceA = nullptr;
  real *DeviceB = nullptr;
  real *DeviceC = nullptr;

  std::vector<real*> DeviceShuffledA{};
  std::vector<real*> DeviceShuffledB{};
  std::vector<real*> DeviceShuffledC{};

  gemmforge::dense::TestDriver Driver;
};

#endif //GEMMS_GEMM_DRIVER_H
