#include "typedef.h"
#include <string>
#include <tuple>
#include <vector>
#include <assert.h>
#include <stdexcept>
#include <iostream>

/*
  The sole change of this driver compared to the simple driver is that it compares Dense x Sparse Matrix multiplication 
  with Dense x Dense already implemented. The correctnesss of this test case relies and assumes that Dense x Dense is already correct.
  Furthermore, it does the test by multiplying both the Dense and Sparse version of B.
*/
namespace gemmforge {
  namespace sparse_dense {
    class TestDriver {
    public:
      TestDriver() {};
      TestDriver(const TestDriver&) = delete;

      TestDriver(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_a_type, bool transA) : 
          m_rowA(rowA), m_colA(colA), 
          m_rowB(rowB), m_colB(colB), 
          m_rowC(rowC), m_colC(colC),
          m_SizeMatA{rowA * colA},
          m_SizeMatB{rowB * colB},
          m_NumElements(NumElements),
          m_matrix_a_type(matrix_a_type),
          m_transA(transA),
          m_IsReady(true){}
      ~TestDriver() {}

      void setParams(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_a_type, bool transA);
      void SetUp(std::string a_type);
      void prepareData(std::string matrix_a_type);
      void TearDown();

      void *getTestStream();

      std::tuple<real*, real*, real*, real*, real*> getDeviceRawData() {
        return std::make_tuple(m_DeviceMatA_dense, m_DeviceMatA_sparse, m_DeviceMatB, m_DeviceMatC1, m_DeviceMatC2);
      }

      std::tuple<real*, real*, real*, real*> getHostRawData() {
        return std::make_tuple(m_HostMatA_dense, m_HostMatA_sparse, m_HostMatB, m_HostMatC);
      }

      std::tuple<real*, real*, real*> getRawResults();

      void retrieveResults(int NumRows,
                             int LeadDim,
                             int NumColumns,
                             int Offset,
                             int Stride,
                             int NumElements,
                             bool sparseResult);

      bool checkEq(real Eps = 1e-5);

      double getDeviceAllocatedMemSize() {
        long long Size = (m_SizeMatA + m_SizeMatB*2 + m_SizeMatC) * m_NumElements * sizeof(real);
        double Factor = 1024 * 1024 * 1024;
        return Size / Factor;
      }

    protected:
      void initMatrix(real *Matrix, int Size);
      void initSparseMatrix(real *DenseVersionOfSparseMatrix, real* SparseMatrix, int rowA, int colA, std::string matrix_a_type);

      float sparsity = 0.25;

      int m_rowA{0};
      int m_colA{0};
      int m_rowB{0};
      int m_colB{0};
      int m_rowC{0};
      int m_colC{0};
      
      int m_SizeMatA{0};
      int m_SizeMatB{0};
      int m_SizeMatC{0};
      int m_NumElements{0};
      std::string m_matrix_a_type{"undefined"};
      bool m_transA{0};

      real *m_HostMatA_dense = nullptr;
      real *m_HostMatA_sparse = nullptr;
      real *m_HostMatB = nullptr;
      real *m_HostMatC = nullptr;
      real *m_ResultsFromDevice1 = nullptr;
      real *m_ResultsFromDevice2 = nullptr;

      real *m_DeviceMatA_dense = nullptr;
      real *m_DeviceMatA_sparse = nullptr;
      real *m_DeviceMatB = nullptr;
      real *m_DeviceMatC1 = nullptr;
      real *m_DeviceMatC2 = nullptr;

      std::string m_Log{};
      bool m_IsReady{false};
      bool m_IsSet{false};

    public:
      int infer_sparse_size(int rowA, int colA, std::string matrix_a_type) {
        if (matrix_a_type == "random"){
          return (rowA*colA)*sparsity;
        }else if (matrix_a_type == "band_diagonal"){
          return -1;
        }else if (matrix_a_type == "single_column"){
          return rowA;
        }else if (matrix_a_type == "single_row"){
          return colA;
        }else if (matrix_a_type == "full") {
          return rowA * colA;
        }else if (matrix_a_type == "chequered") {
          return -1;
        }else {
          throw std::runtime_error("Unallowed Matrix B Type!");
        }
      }

      int _2d21d(int i, int j, int rowX, int colX, int Element, bool col_major = true){
        if (col_major){
          return Element*rowX*colX + i + j*rowX;
        }else{
          return Element*rowX*colX + i*colX + j;
        }
      }
    };
  }
}