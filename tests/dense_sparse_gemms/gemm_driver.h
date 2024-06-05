#ifndef GEMMS_GEMM_DRIVER_H
#define GEMMS_GEMM_DRIVER_H

#include "typedef.h"
#include "simple_dense_sparse_driver.h"
#include "gtest/gtest.h"
#include <tuple>
#include <vector>

class DenseXSparseGemmTest : public ::testing::Test
{
protected:
  void SetUp(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_b_type, bool transB)
  {
    Driver.setParams(rowA, colA, rowB, colB, rowC, colC, NumElements, matrix_b_type, transB);
    Driver.SetUp(matrix_b_type);

    std::tie(HostA, HostB_dense, HostB_sparse, HostC) = Driver.getHostRawData();
    std::tie(DeviceA, DeviceB_dense, DeviceB_sparse, DeviceC1, DeviceC2) = Driver.getDeviceRawData();
    // std::tie(DeviceShuffledA, DeviceShuffledB, DeviceShuffledC) = Driver.getShuffledDeviceData();
  }

  void TearDown()
  {
    Driver.TearDown();
  }

  real *HostA = nullptr;
  real *HostB_dense = nullptr;
  real *HostB_sparse = nullptr;
  real *HostC = nullptr;

  real *DeviceA = nullptr;
  real *DeviceB_dense = nullptr;
  real *DeviceB_sparse = nullptr;
  real *DeviceC1 = nullptr;
  real *DeviceC2 = nullptr;

  // std::vector<real*> DeviceShuffledA{};
  // std::vector<real*> DeviceShuffledB{};
  // std::vector<real*> DeviceShuffledC{};

  tensorforge::dense_sparse::TestDriver Driver;
};

#endif // GEMMS_GEMM_DRIVER_H
