#include "typedef.h"
#include <string>
#include <tuple>
#include <vector>
#include <assert.h>

namespace gemmforge {
  namespace dense {

    class TestDriver {
    public:
      TestDriver() {};
      TestDriver(const TestDriver&) = delete;

      TestDriver(int SizeMatA, int SizeMatB, int SizeMatC, int NumElements) : m_SizeMatA(SizeMatA),
                                                                              m_SizeMatB{SizeMatB},
                                                                              m_SizeMatC(SizeMatC),
                                                                              m_NumElements(NumElements),
                                                                              m_IsReady(true){}
      ~TestDriver() {}

      void setParams(int SizeMatA, int SizeMatB, int SizeMatC, int NumElements);
      void SetUp();
      void prepareData();
      void TearDown();

      void *getTestStream();

      std::tuple<real*, real*, real*> getDeviceRawData() {
        return std::make_tuple(m_DeviceMatA, m_DeviceMatB, m_DeviceMatC);
      }

      std::tuple<real*, real*, real*> getHostRawData() {
        return std::make_tuple(m_HostMatA, m_HostMatB, m_HostMatC);
      }

      std::tuple<std::vector<real*>, std::vector<real*>, std::vector<real*>> getShuffledDeviceData();
      std::tuple<real*, real*> getRawResults();

      void packResults(int NumRows, int LeadDim, int NumColumns, int Offset, int Stride, int NumElements);
      std::tuple<PackedData, PackedData> getPackedResults() {
        return std::make_tuple(m_PackedHostResults, m_PackedDeviceResults);
      }

      template <class ComparatorType>
      bool isTestPassed(real Eps = 1e-5) {
        assert(!m_PackedHostResults.empty() != 0 && "Host results has not been packed");
        assert(!m_PackedDeviceResults.empty() && "Device results has not been packed");
        ComparatorType Comparator;
        return Comparator.compare(m_PackedHostResults, m_PackedDeviceResults, Eps);
      }

      template <class ComparatorType>
      bool isTestPassed(ComparatorType &Comparator, real Eps = 1e-5) {
        assert(!m_PackedHostResults.empty() != 0 && "Host results has not been packed");
        assert(!m_PackedDeviceResults.empty() && "Device results has not been packed");
        return Comparator.compare(m_PackedHostResults, m_PackedDeviceResults, Eps);
      }

      double getDeviceAllocatedMemSize() {
        long long Size = (m_SizeMatA + m_SizeMatB + m_SizeMatC) * m_NumElements * sizeof(real);
        double Factor = 1024 * 1024 * 1024;
        return Size / Factor;
      }

    protected:
      void initMatrix(real *Matrix, int Size);

      int m_SizeMatA{};
      int m_SizeMatB{};
      int m_SizeMatC{};
      int m_NumElements{};

      real *m_HostMatA = nullptr;
      real *m_HostMatB = nullptr;
      real *m_HostMatC = nullptr;
      real *m_ResultsFromDevice = nullptr;

      real *m_DeviceMatA = nullptr;
      real *m_DeviceMatB = nullptr;
      real *m_DeviceMatC = nullptr;

      PackedData m_PackedHostResults{};
      PackedData m_PackedDeviceResults{};

      std::string m_Log{};
      bool m_IsReady{false};
      bool m_IsSet{false};

    };
  }
}