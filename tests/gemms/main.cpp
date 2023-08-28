#include "gtest/gtest.h"
#include "device.h"
#include <iostream>

using namespace device;

int main(int Argc, char *Argv[]) {
  ::testing::InitGoogleTest(&Argc, Argv);
  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(0);
  device.api->initialize();

  return RUN_ALL_TESTS();
}
