
#pragma once

#include <string>
#include "ISvm.h"

namespace phd { namespace svm
{
KernelTypes stringToKernelType(std::string& kernelName);

std::string kernelTypeToString(KernelTypes kernelType);

std::string svmTypeToString(SvmTypes svmType);
}}// namespace phd::svm
