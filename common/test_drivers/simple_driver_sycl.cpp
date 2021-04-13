#include "aux.h"
#include "simple_driver.h"
#include "gemmgen_aux.h"
#include <sstream>
#include <iostream>
#include <CL/sycl.hpp>

using namespace gemmgen::dense;

cl::sycl::queue *testqueue;

void TestDriver::setParams(int SizeMatA, int SizeMatB, int SizeMatC, int NumElements) {
    m_NumElements = NumElements;

    m_SizeMatA = SizeMatA;
    m_SizeMatB = SizeMatB;
    m_SizeMatC = SizeMatC;
    m_IsReady = true;
}

void TestDriver::SetUp() {
    if (m_IsReady && !m_IsSet) {
        m_HostMatA = new real[m_SizeMatA * m_NumElements];
        m_HostMatB = new real[m_SizeMatB * m_NumElements];
        m_HostMatC = new real[m_SizeMatC * m_NumElements];
        m_ResultsFromDevice = new real[m_SizeMatC * m_NumElements];

        testqueue = new cl::sycl::queue{cl::sycl::gpu_selector{}, cl::sycl::property::queue::in_order()};

        this->m_DeviceMatA = (real *)cl::sycl::malloc_device(m_SizeMatA * m_NumElements * sizeof(real), *testqueue);
        this->m_DeviceMatB = (real *)cl::sycl::malloc_device(m_SizeMatB * m_NumElements * sizeof(real), *testqueue);
        this->m_DeviceMatC = (real *)cl::sycl::malloc_device(m_SizeMatC * m_NumElements * sizeof(real), *testqueue);

        testqueue->wait_and_throw();

        m_IsSet = true;

        prepareData();
    } else if (!m_IsReady) {
        throw std::string("Test hasn't been parameterized. Use a Parameterized Constructor or SetParams method.");
    } else if (m_IsSet) {
        throw std::string("you tied to SetUp a test the second time. TearDown the test first");
    }
}

void *TestDriver::getTestStream() {
    return testqueue;
}

void TestDriver::prepareData() {
    if (m_IsSet) {
        initMatrix(m_HostMatA, m_SizeMatA);
        initMatrix(m_HostMatB, m_SizeMatB);
        initMatrix(m_HostMatC, m_SizeMatC);

        testqueue->memcpy(m_DeviceMatA, m_HostMatA, m_SizeMatA * m_NumElements * sizeof(real));
        testqueue->memcpy(m_DeviceMatB, m_HostMatB, m_SizeMatB * m_NumElements * sizeof(real));
        testqueue->memcpy(m_DeviceMatC, m_HostMatC, m_SizeMatC * m_NumElements * sizeof(real));
        testqueue->wait_and_throw();

    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
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


std::tuple<std::vector<real *>, std::vector<real *>, std::vector<real *>> TestDriver::getShuffledDeviceData() {
    if (m_IsSet) {
        return std::make_tuple(shuffleMatrices(m_DeviceMatA, m_SizeMatA, m_NumElements),
                               shuffleMatrices(m_DeviceMatB, m_SizeMatB, m_NumElements),
                               shuffleMatrices(m_DeviceMatC, m_SizeMatC, m_NumElements));
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}


std::tuple<real *, real *> TestDriver::getRawResults() {
    if (m_IsSet) {
        testqueue->memcpy(m_ResultsFromDevice, m_DeviceMatC, m_SizeMatC * m_NumElements * sizeof(real)).wait_and_throw();
        return std::make_tuple(m_HostMatC, m_ResultsFromDevice);
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}


void TestDriver::packResults(int NumRows,
                             int LeadDim,
                             int NumColumns,
                             int Offset,
                             int Stride,
                             int NumElements) {
    if (m_IsSet) {
        testqueue->memcpy(m_ResultsFromDevice, m_DeviceMatC, m_SizeMatC * m_NumElements * sizeof(real)).wait_and_throw();

        m_PackedHostResults.clear();
        m_PackedHostResults.resize(NumElements);

        m_PackedDeviceResults.clear();
        m_PackedDeviceResults.resize(NumElements);


        real *HostRawData = &m_HostMatC[Offset];
        real *DeviceRawData = &m_ResultsFromDevice[Offset];

        const int Size = NumRows * NumColumns;
        for (int Element = 0; Element < NumElements; ++Element) {
            std::vector<real> ElementFromHost(Size, 0.0);

            for (int Col = 0; Col < NumColumns; ++Col) {
                for (int Row = 0; Row < NumRows; ++Row) {
                    const int Index = Row + Col * NumRows;
                    ElementFromHost[Index] = HostRawData[Row + Col * LeadDim + Stride * Element];
                }
            }
            m_PackedHostResults[Element] = std::move(ElementFromHost);


            std::vector<real> ElementFromDevice(Size, 0.0);
            for (int Col = 0; Col < NumColumns; ++Col) {
                for (int Row = 0; Row < NumRows; ++Row) {
                    const int Index = Row + Col * NumRows;
                    ElementFromDevice[Index] = DeviceRawData[Row + Col * LeadDim + Stride * Element];
                }
            }
            m_PackedDeviceResults[Element] = std::move(ElementFromDevice);
        }
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}


void TestDriver::TearDown() {
    if (m_IsSet) {
        delete[] m_HostMatA;
        delete[] m_HostMatB;
        delete[] m_HostMatC;
        delete[] m_ResultsFromDevice;

        cl::sycl::free(m_DeviceMatA, *testqueue);
        cl::sycl::free(m_DeviceMatB, *testqueue);
        cl::sycl::free(m_DeviceMatC, *testqueue);

        delete testqueue;

        m_IsSet = false;
    } else {
        throw std::string("Test hasn't been set. Call SetUp method first");
    }
}
