#include "aux.h"
#include "simple_sparse_dense_driver.h"
#include "kernelforge_aux.h"
#include <sstream>
#include <iostream>
#include <device.h>
#include <random>
#include <vector>
#include <tuple>

using namespace kernelforge::sparse_dense;
using namespace device;

extern std::vector<std::tuple<int, int>> get_coordinates_A_core();

AbstractAPI * getDeviceAPI2() {
    auto device = &DeviceInstance::getInstance();
    return device->api;
}

void TestDriver::setParams(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_a_type, bool transA) {
    m_NumElements = NumElements;
    m_SizeMatA = rowA * colA;
    m_SizeMatB = rowB * colB;
    m_SizeMatC = rowC * colC;
    m_rowA = rowA;
    m_colA = colA;
    m_rowB = rowB;
    m_colB = colB;
    m_rowC = rowC;
    m_colC = colC;
    m_IsReady = true;
    m_matrix_a_type = matrix_a_type;
    m_transA = transA;
}

void TestDriver::SetUp(std::string b_type) {
    if (m_IsReady && !m_IsSet) {
        m_HostMatA_dense = new real[m_SizeMatA * m_NumElements]{0};
        m_HostMatA_sparse = new real[infer_sparse_size(m_rowA, m_colA, m_matrix_a_type) * m_NumElements]{0};
        m_HostMatB = new real[m_SizeMatB * m_NumElements]{0};
        m_HostMatC = new real[m_SizeMatC * m_NumElements]{0};
        m_ResultsFromDevice1 = new real[m_SizeMatC * m_NumElements]{0};
        m_ResultsFromDevice2 = new real[m_SizeMatC * m_NumElements]{0};

        m_DeviceMatA_dense = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatA * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatA_sparse = (real *) getDeviceAPI2()->allocGlobMem(infer_sparse_size(m_rowA, m_colA, m_matrix_a_type) * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatB = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatB * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatC1 = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatC2 = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;

        m_IsSet = true;

    } else if (!m_IsReady) {
        throw std::string("Test hasn't been parameterized. Use a Parameterized Constructor or SetParams method.");
    } else if (m_IsSet) {
        throw std::string("you tied to SetUp a test the second time. TearDown the test first");
    }
}


void TestDriver::prepareData(std::string a_type) {
    if (m_IsSet) {
    initSparseMatrix(m_HostMatA_dense, m_HostMatA_sparse, m_rowA, m_colA, a_type);
        initMatrix(m_HostMatB, m_SizeMatB);
        initMatrix(m_HostMatC, m_SizeMatC);

        getDeviceAPI2()->copyTo(m_DeviceMatA_dense, m_HostMatA_dense, m_SizeMatA * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyTo(m_DeviceMatA_sparse, m_HostMatA_sparse, infer_sparse_size(m_rowA, m_colA, a_type) * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyTo(m_DeviceMatB, m_HostMatB, m_SizeMatB * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyTo(m_DeviceMatC1, m_HostMatC, m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyTo(m_DeviceMatC2, m_HostMatC, m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}

void *TestDriver::getTestStream() {
    return getDeviceAPI2()->getDefaultStream();
}

void TestDriver::initMatrix(real *Matrix, int Size) {
    if (m_IsSet) {
        for (int Element = 0; Element < m_NumElements; ++Element) {
            for (int Index = 0; Index < Size; ++Index) {
                Matrix[Index + Size * Element] = getRandomNumber();
                //Matrix[Index + Size * Element] = 1.1;
            }
        }
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}


void TestDriver::initSparseMatrix(real *DenseVersionOfSparseMatrix, real* SparseMatrix, int rowA, int colA, std::string matrix_a_type) {
    if (!m_IsSet){
        throw std::string("Test hasn't been set. Call SetUp method first");
    }

    for (int i = 0; i < m_NumElements*rowA*colA; i++){
        DenseVersionOfSparseMatrix[i] = 0.0; 
    }
    
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<real> dist(1.0, 100.0);

    if (matrix_a_type == "band_diagonal"){
    }else if (matrix_a_type == "single_column"){
    }else if (matrix_a_type == "single_row"){
    }else if (matrix_a_type == "chequered"){
    }else if (matrix_a_type == "full"){
        if (!m_transA){
            int iter = 0;
            real a = 0.0;
            for (int Element = 0; Element < m_NumElements; Element++){
                for (int i = 0; i < rowA; i++){
                    for (int j = 0; j < colA; j++){
                        a = dist(mt);
                        DenseVersionOfSparseMatrix[_2d21d(i, j, rowA, colA, Element, !m_transA)] = a;
                        SparseMatrix[_2d21d(i, j, rowA, colA, Element, !m_transA)] = a;
                    }
                }
            }
        }else{
            int iter = 0;
            real a = 0.0;
            for (int Element = 0; Element < m_NumElements; Element++){
                for (int i = 0; i < rowA; i++){
                    for (int j = 0; j < colA; j++){
                        a = dist(mt);
                        DenseVersionOfSparseMatrix[iter] = a;
                        SparseMatrix[iter++] = a;
                    }
                }
            }
        }
    }else if (matrix_a_type == "random"){
        std::vector<std::tuple<int, int>> coordinates = get_coordinates_A_core();
        int iter = 0;
        real a = 0.0;
        for (int Element = 0; Element < m_NumElements; Element++){
            for (auto& coordinate_pair : coordinates){
                int i = std::get<0>(coordinate_pair);
                int j = std::get<1>(coordinate_pair);
                a = dist(mt);
                DenseVersionOfSparseMatrix[_2d21d(i, j, rowA, colA, Element, true)] = a;
                SparseMatrix[iter++] = a;
            }
        }
    }else {
        std::runtime_error("matrix_b_type needs to be full | random");
    }
}



std::tuple<real *, real *, real*> TestDriver::getRawResults() {
    if (m_IsSet) {
        getDeviceAPI2()->copyFrom(m_ResultsFromDevice1, m_DeviceMatC1, m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyFrom(m_ResultsFromDevice2, m_DeviceMatC2, m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;
        return std::make_tuple(m_HostMatC, m_ResultsFromDevice1, m_ResultsFromDevice2);
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}

void TestDriver::retrieveResults(int NumRows,
                             int LeadDim,
                             int NumColumns,
                             int Offset,
                             int Stride,
                             int NumElements,
                             bool sparseResult) {
    if (m_IsSet) {
        if (!sparseResult){
            getDeviceAPI2()->copyFrom(m_ResultsFromDevice1, m_DeviceMatC1, m_SizeMatC * m_NumElements * sizeof(real));
            CHECK_ERR;
        } else {
            getDeviceAPI2()->copyFrom(m_ResultsFromDevice2, m_DeviceMatC2, m_SizeMatC * m_NumElements * sizeof(real));
            CHECK_ERR;
        }
    }
}


bool TestDriver::checkEq(real Eps){
    for (int i = 0; i < m_rowC*m_colC*m_NumElements; i++) {
        if (std::abs(m_ResultsFromDevice1[i] - m_ResultsFromDevice2[i]) > Eps){
            real* R1 = m_ResultsFromDevice1;
            real* R2 = m_ResultsFromDevice2;
            std::cout << "[";
            for (int ii = 0; ii < 56*9 -1; ii++){
                std::cout << R1[ii] << ", ";
            }
            std::cout << R1[56*9 -1] << "]" << std::endl;
            std::cout << "[";
            for (int ii = 0; ii < 56*9 - 1; ii++){
                std::cout << R2[ii] << ", ";
            }
            std::cout << R2[56*9 -1] << "]" << std::endl;     
            return false;
        }
    }
    return true;
}

void TestDriver::TearDown() {
    if (m_IsSet) {
        delete[] m_HostMatA_dense; m_HostMatA_dense = nullptr;
        delete[] m_HostMatA_sparse; m_HostMatA_sparse = nullptr;
        delete[] m_HostMatB;  m_HostMatB = nullptr;
        delete[] m_HostMatC;    m_HostMatC;
        delete[] m_ResultsFromDevice1;  m_ResultsFromDevice1 = nullptr;
        delete[] m_ResultsFromDevice2;  m_ResultsFromDevice2 = nullptr;

        getDeviceAPI2()->freeMem(m_DeviceMatA_dense); m_DeviceMatA_dense = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatA_sparse);   m_DeviceMatA_sparse = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatB);  m_DeviceMatB = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatC1);    m_DeviceMatC1 = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatC2);    m_DeviceMatC2 = nullptr;
        m_IsSet = false;
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}
