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
namespace kernelforge {
  namespace dense_sparse {
    class TestDriver {
    public:
      TestDriver() {};
      TestDriver(const TestDriver&) = delete;

      TestDriver(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_b_type, bool transB) : 
          m_rowA(rowA), m_colA(colA), 
          m_rowB(rowB), m_colB(colB), 
          m_rowC(rowC), m_colC(colC),
          m_SizeMatA{rowA * colA},
          m_SizeMatB{rowB * colB},
          m_NumElements(NumElements),
          m_matrix_b_type(matrix_b_type),
          m_transB(transB),
          m_IsReady(true){}
      ~TestDriver() {}

      void setParams(int rowA, int colA, int rowB, int colB, int rowC, int colC, int NumElements, std::string matrix_b_type, bool transB);
      void SetUp(std::string b_type);
      void prepareData(std::string matrix_b_type);
      void TearDown();

      void *getTestStream();

      std::tuple<real*, real*, real*, real*, real*> getDeviceRawData() {
        return std::make_tuple(m_DeviceMatA, m_DeviceMatB_dense, m_DeviceMatB_sparse, m_DeviceMatC1, m_DeviceMatC2);
      }

      std::tuple<real*, real*, real*, real*> getHostRawData() {
        return std::make_tuple(m_HostMatA, m_HostMatB_dense, m_HostMatB_sparse, m_HostMatC);
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
      void initSparseMatrix(real *DenseVersionOfSparseMatrix, real* SparseMatrix, int rowB, int colB, std::string matrix_b_type);

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
      std::string m_matrix_b_type{"undefined"};
      bool m_transB{0};

      real *m_HostMatA = nullptr;
      real *m_HostMatB_dense = nullptr;
      real *m_HostMatB_sparse = nullptr;
      real *m_HostMatC = nullptr;
      real *m_ResultsFromDevice1 = nullptr;
      real *m_ResultsFromDevice2 = nullptr;

      real *m_DeviceMatA = nullptr;
      real *m_DeviceMatB_dense = nullptr;
      real *m_DeviceMatB_sparse = nullptr;
      real *m_DeviceMatC1 = nullptr;
      real *m_DeviceMatC2 = nullptr;

      std::string m_Log{};
      bool m_IsReady{false};
      bool m_IsSet{false};

    public:
      int infer_sparse_size(int rowB, int colB, std::string matrix_b_type) {
        if (matrix_b_type == "random"){
          return (rowB*colB)*sparsity;
        }else if (matrix_b_type == "band_diagonal"){
          return 2 + 2 + (rowB-2)*3;
        }else if (matrix_b_type == "single_column"){
          return rowB;
        }else if (matrix_b_type == "single_row"){
          return colB;
        }else if (matrix_b_type == "full") {
          return rowB * colB;
        }else if (matrix_b_type == "chequered") {
          if (colB % 2 == 0){
            return static_cast<int>(rowB * colB / 2);
          }else{
            // 1 row A+1, 1 row A elements
            // Where A == L colB / 2 (rounded down to nearest int)
            // This means colB many white elements per 2 rows
            if (rowB % 2 == 0){
              return static_cast<int>(colB * rowB / 2);
            } else {
              // Implementation specific of the element orders, in my case first row has +1 element,
              // when unequal then we get 1 too many
              return 1 + static_cast<int>(colB * rowB / 2);
            }
          }
        }else {
          throw std::runtime_error("Unallowed Matrix B Type!");
        }
      }

      int _2d21d(int i, int j, int rowX, int colX, int Element){
        return Element*rowX*colX + i + j*rowX;
      }
    };
  }
}