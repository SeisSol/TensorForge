#include "gtest/gtest.h"
#include <iostream>

int main(int Argc, char *Argv[]) {
  ::testing::InitGoogleTest(&Argc, Argv);
  return RUN_ALL_TESTS();
}