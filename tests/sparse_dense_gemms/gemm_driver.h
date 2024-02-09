#pragma once

#include "typedef.h"
#include "simple_sparse_dense_driver.h"
#include "gtest/gtest.h"
#include <tuple>
#include <vector>

class SparseXDenseGemmTest : public ::testing::Test {
protected:
  void SetUp(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_a_type, bool transA) {
    Driver.setParams(rowA, colA, rowB, colB, rowC, colC, NumElements, matrix_a_type, transA);
    Driver.SetUp(matrix_a_type);

    std::tie(HostA_dense, HostA_sparse, HostB,  HostC) = Driver.getHostRawData();
    std::tie(DeviceA_dense, DeviceA_sparse, DeviceB, DeviceC1, DeviceC2) = Driver.getDeviceRawData();
    //std::tie(DeviceShuffledA, DeviceShuffledB, DeviceShuffledC) = Driver.getShuffledDeviceData();
  }

  void TearDown() {
    Driver.TearDown();
  }

  real *HostA_dense = nullptr;
  real *HostA_sparse = nullptr;
  real *HostB = nullptr;
  real *HostC = nullptr;

  real *DeviceA_dense = nullptr;
  real *DeviceA_sparse = nullptr;
  real *DeviceB = nullptr;
  real *DeviceC1 = nullptr;
  real *DeviceC2 = nullptr;

  //std::vector<real*> DeviceShuffledA{};
  //std::vector<real*> DeviceShuffledB{};
  //std::vector<real*> DeviceShuffledC{};

  kernelforge::sparse_dense::TestDriver Driver;
};
