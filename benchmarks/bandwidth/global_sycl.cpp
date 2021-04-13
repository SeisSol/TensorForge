#include "stop_watch.h"
#include "gemmgen_aux.h"
#include "stop_watch.h"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <CL/sycl.hpp>

using namespace gemmgen;

void copyData(float *To, float *From, size_t size, cl::sycl::range<1> global, cl::sycl::range<1> local, cl::sycl::queue *stream) {
  stream->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(cl::sycl::nd_range<1>{global, local}, [=](cl::sycl::nd_item<1> item) {
          if (item.get_global_id(0) < size) {
            To[item.get_global_id(0)] = From[item.get_global_id(0)];
          }
      });
  });
}


int main(int Argc, char *Argv[]) {
  YAML::Node Config = YAML::LoadFile("../config.yaml");
  int NumRepeats = Config["num_repeats"].as<int>();
  double AllocatedMemGb = Config["allocated_mem"].as<double>();

  constexpr long long FACTOR = 1024 * 1024 * 1024;
  size_t NumElements = (FACTOR * AllocatedMemGb) / sizeof(float);

  float *To = nullptr;
  float *From = nullptr;

  cl::sycl::queue q{cl::sycl::gpu_selector{}, cl::sycl::property::queue::in_order()};

  To = cl::sycl::malloc_device<float>(NumElements, q);
  From = cl::sycl::malloc_device<float>(NumElements, q);

  size_t workItemSize = q.get_device().get_info<cl::sycl::info::device::max_work_item_sizes>()[0];
  std::cout << "NOTE: max work items per group is " << workItemSize << std::endl;
  cl::sycl::range<1> global{((NumElements + 1024 - 1) / 1024 ) * 1024};
  cl::sycl::range<1> local{workItemSize};

  utils::StopWatch<std::chrono::duration<double, std::chrono::nanoseconds::period>> Timer;
  Timer.start();
  for (int Repeat = 0; Repeat < NumRepeats; ++Repeat) {
    copyData(To, From, NumElements, global, local, &q);
  }
  q.wait();
  Timer.stop();

  auto AverageTime = Timer.getTime() / NumRepeats;
  // 1 copy and 1 write explains the factor of 2
  double BandwidthGb = 2 * (NumElements / AverageTime) * sizeof(float);
  std::cout << "Allocated Mem, GB: " << AllocatedMemGb << std::endl;
  std::cout << "Time: " << Timer.getTime() << std::endl;
  std::cout << "Num. Repeats: " << NumRepeats << std::endl;
  std::cout << "Num. Elements: " << NumElements << std::endl;
  std::cout << "Achieved bandwidth: " << BandwidthGb << " GB/s" << std::endl;

  cl::sycl::free(To, q);
  cl::sycl::free(From, q);

  return 0;
}
