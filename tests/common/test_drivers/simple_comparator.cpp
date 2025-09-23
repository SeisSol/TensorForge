#include "comparators.h"
#include <iostream>
#include <sstream>

bool L1NormComparator::compare(const PackedData &Host, const PackedData &Device, real Eps) {
  bool IsEqual = true;

  std::stringstream Stream{};
  const unsigned NumElements = Host.size();
  for (int Element = 0; Element < NumElements; ++Element) {

    const unsigned Size = Host[Element].size();
    for (int Index = 0; Index < Size; ++Index) {
      real Difference = fabs(Host[Element][Index] - Device[Element][Index]);
      if (Difference > Eps) {
        IsEqual = false;

        Stream << "Element: " << Element << "; " << "Index: " << Index << "; "
               << "Host: " << Host[Element][Index] << "; "
               << "Device: " << Device[Element][Index] << "; "
               << "Diff.: " << Difference << "\n";

      }
    }
  }

  if (!IsEqual) {
    std::cout << Stream.str() << std::endl;
  }

  return IsEqual;
}

