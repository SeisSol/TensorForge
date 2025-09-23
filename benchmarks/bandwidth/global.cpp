#include "stop_watch.h"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <device.h>
#include "kernel.h"

using namespace device;


int main(int Argc, char *Argv[]) {
    DeviceInstance &device = DeviceInstance::getInstance();
    device.api->setDevice(0);
    device.api->initialize();

    YAML::Node Config = YAML::LoadFile("../config.yaml");
    int NumRepeats = Config["num_repeats"].as<int>();
    auto AllocatedMemGb = Config["allocated_mem"].as<double>();

    constexpr long long FACTOR = 1024 * 1024 * 1024;
    size_t NumElements = (FACTOR * AllocatedMemGb) / sizeof(float);

    auto To = (float *) device.api->allocGlobMem(NumElements * sizeof(float));;
    auto From = (float *) device.api->allocGlobMem(NumElements * sizeof(float));;

    size_t blocks = (NumElements + 1024 - 1) / 1024;
    size_t threads = 1024;
    device.api->synchDevice();

    utils::StopWatch <std::chrono::duration<double, std::chrono::nanoseconds::period>> Timer;
    Timer.start();
    for (int Repeat = 0; Repeat < NumRepeats; ++Repeat) {
        copyData(To, From, NumElements, blocks, threads, device.api->getDefaultStream());
    }
    device.api->synchDevice();
    Timer.stop();

    auto AverageTime = Timer.getTime() / NumRepeats;
    // 1 copy and 1 write explains the factor of 2
    double BandwidthGb = 2 * (NumElements / AverageTime) * sizeof(float);
    std::cout << "Allocated Mem, GB: " << AllocatedMemGb << std::endl;
    std::cout << "Time: " << Timer.getTime() << std::endl;
    std::cout << "Num. Repeats: " << NumRepeats << std::endl;
    std::cout << "Num. Elements: " << NumElements << std::endl;
    std::cout << "Achieved bandwidth: " << BandwidthGb << " GB/s" << std::endl;

    device.api->freeMem(To);
    device.api->freeMem(From);

    device.api->finalize();

    return 0;
}
