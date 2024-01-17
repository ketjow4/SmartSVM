
#pragma once

#include "SvmLib/ISvm.h"

namespace genetic
{
class ISvmAlgorithm
{
public:
    virtual ~ISvmAlgorithm() = default;

    std::shared_ptr<phd::svm::ISvm> virtual run() = 0;

    virtual void setC(double /*C*/)  {};
};
} // namespace genetic