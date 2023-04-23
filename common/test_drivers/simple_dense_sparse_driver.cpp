#include "aux.h"
#include "simple_dense_sparse_driver.h"
#include "gemmforge_aux.h"
#include <sstream>
#include <iostream>
#include <device.h>

using namespace gemmforge::dense_sparse;
using namespace device;

AbstractAPI * getDeviceAPI2() {
    auto device = &DeviceInstance::getInstance();
    return device->api;
}

void TestDriver::setParams(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_b_type, bool transB) {
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
    m_matrix_b_type = matrix_b_type;
    m_transB = transB;
}

void TestDriver::SetUp(std::string b_type) {
    if (m_IsReady && !m_IsSet) {
        m_HostMatA = new real[m_SizeMatA * m_NumElements]{0};
        m_HostMatB_dense = new real[m_SizeMatB * m_NumElements]{0};
        m_HostMatB_sparse = new real[infer_sparse_size(m_rowB, m_colB, m_matrix_b_type) * m_NumElements]{0};
        m_HostMatC = new real[m_SizeMatC * m_NumElements]{0};
        m_ResultsFromDevice1 = new real[m_SizeMatC * m_NumElements]{0};
        m_ResultsFromDevice2 = new real[m_SizeMatC * m_NumElements]{0};

        m_DeviceMatA = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatA * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatB_dense = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatB * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatB_sparse = (real *) getDeviceAPI2()->allocGlobMem(infer_sparse_size(m_rowB, m_colB, m_matrix_b_type) * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatC1 = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;
        m_DeviceMatC2 = (real *) getDeviceAPI2()->allocGlobMem(m_SizeMatC * m_NumElements * sizeof(real));
        CHECK_ERR;

        m_IsSet = true;

        prepareData(b_type);
    } else if (!m_IsReady) {
        throw std::string("Test hasn't been parameterized. Use a Parameterized Constructor or SetParams method.");
    } else if (m_IsSet) {
        throw std::string("you tied to SetUp a test the second time. TearDown the test first");
    }
}


void TestDriver::prepareData(std::string b_type) {
    if (m_IsSet) {
        initMatrix(m_HostMatA, m_SizeMatA);
        initSparseMatrix(m_HostMatB_dense, m_HostMatB_sparse, m_rowB, m_colB, b_type);
        initMatrix(m_HostMatC, m_SizeMatC);

        getDeviceAPI2()->copyTo(m_DeviceMatA, m_HostMatA, m_SizeMatA * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyTo(m_DeviceMatB_dense, m_HostMatB_dense, m_SizeMatB * m_NumElements * sizeof(real));
        CHECK_ERR;
        getDeviceAPI2()->copyTo(m_DeviceMatB_sparse, m_HostMatB_sparse, infer_sparse_size(m_rowB, m_colB, m_matrix_b_type) * m_NumElements * sizeof(real));
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
            }
        }
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}


void TestDriver::initSparseMatrix(real *DenseVersionOfSparseMatrix, real* SparseMatrix, int rowB, int colB, std::string matrix_b_type) {
    if (!m_IsSet){
        throw std::string("Test hasn't been set. Call SetUp method first");
    }

    if (matrix_b_type == "band_diagonal"){
        if (!m_transB){
            int iter = 0;
            for (int Element = 0; Element < m_NumElements; Element++){
                DenseVersionOfSparseMatrix[_2d21d(0, 0, rowB, colB, Element, !m_transB)] = 2.0;
                DenseVersionOfSparseMatrix[_2d21d(1, 0, rowB, colB, Element, !m_transB)] = 1.0;
                SparseMatrix[iter++] = 2.0;
                SparseMatrix[iter++] = 1.0;
                
                for (int j = 1; j < colB - 1; j++){
                    DenseVersionOfSparseMatrix[_2d21d(j-1, j, rowB, colB, Element, !m_transB)] = 3.0;
                    DenseVersionOfSparseMatrix[_2d21d(j, j, rowB, colB, Element, !m_transB)] = 2.0;
                    DenseVersionOfSparseMatrix[_2d21d(j+1, j, rowB, colB, Element, !m_transB)] = 1.0;
                    SparseMatrix[iter++] = 3.0;
                    SparseMatrix[iter++] = 2.0;
                    SparseMatrix[iter++] = 1.0;
                }

                int j = colB - 1;
                DenseVersionOfSparseMatrix[_2d21d(j-1, j, rowB, colB, Element, !m_transB)] = 3.0;
                DenseVersionOfSparseMatrix[_2d21d(j, j, rowB, colB, Element, !m_transB)] = 2.0;
                SparseMatrix[iter++] = 3.0;
                SparseMatrix[iter++] = 2.0;
            }
        }else{
            int iter = 0;
            for (int Element = 0; Element < m_NumElements; Element++){
                DenseVersionOfSparseMatrix[_2d21d(0, 0, rowB, colB, Element, m_transB)] = 2.0;
                DenseVersionOfSparseMatrix[_2d21d(0, 1, rowB, colB, Element, m_transB)] = 3.0;
                SparseMatrix[iter++] = 2.0;
                SparseMatrix[iter++] = 3.0;
                
                for (int i = 1; i < rowB - 1; i++){
                    DenseVersionOfSparseMatrix[_2d21d(i, i-1, rowB, colB, Element, m_transB)] = 1.0;
                    DenseVersionOfSparseMatrix[_2d21d(i, i, rowB, colB, Element, m_transB)] = 2.0;
                    DenseVersionOfSparseMatrix[_2d21d(i, i+1, rowB, colB, Element, m_transB)] = 3.0;
                    SparseMatrix[iter++] = 1.0;
                    SparseMatrix[iter++] = 2.0;
                    SparseMatrix[iter++] = 3.0;
                }

                int i = rowB - 1;
                DenseVersionOfSparseMatrix[_2d21d(i, i, rowB, colB, Element, m_transB)] = 2.0;
                DenseVersionOfSparseMatrix[_2d21d(i, i+1, rowB, colB, Element, m_transB)] = 3.0;
                SparseMatrix[iter++] = 2.0;
                SparseMatrix[iter++] = 3.0;
            }
        }
    }else if (matrix_b_type == "one_column"){
        std::runtime_error("TODO");
    }else if (matrix_b_type == "one_row"){
        std::runtime_error("TODO");
    }else if (matrix_b_type == "random_entries"){
        std::runtime_error("TODO");
    }else {
        std::runtime_error("matrix_b_type needs to be band | one_column | one_row | random_entries");
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
            return false;
        }
    }
    return true;
}

void TestDriver::TearDown() {
    if (m_IsSet) {
        delete[] m_HostMatA; m_HostMatA = nullptr;
        delete[] m_HostMatB_dense;  m_HostMatB_dense = nullptr;
        delete[] m_HostMatB_sparse; m_HostMatB_sparse = nullptr;
        delete[] m_HostMatC;    m_HostMatC;
        delete[] m_ResultsFromDevice1;  m_ResultsFromDevice1 = nullptr;
        delete[] m_ResultsFromDevice2;  m_ResultsFromDevice2 = nullptr;

        getDeviceAPI2()->freeMem(m_DeviceMatA); m_DeviceMatA = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatB_dense);   m_DeviceMatB_dense = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatC1);    m_DeviceMatC1 = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatB_sparse);  m_DeviceMatB_sparse = nullptr;
        getDeviceAPI2()->freeMem(m_DeviceMatC2);    m_DeviceMatC2 = nullptr;
        m_IsSet = false;
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}
