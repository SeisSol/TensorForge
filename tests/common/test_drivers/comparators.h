#ifndef GEMMS_COMPARATORS_H
#define GEMMS_COMPARATORS_H

#include "typedef.h"
#include <math.h>

using SimpleComparator = class L1NormComparator;

class L1NormComparator {
public:
  bool compare(const PackedData &Host, const PackedData &Device, real Eps = 1e-5);
};


class GoogleComparator {
public:
  bool compare(const PackedData &Host, const PackedData &Device);
};

#endif //GEMMS_COMPARATORS_H
