
#pragma once

#include <gsl/gsl>
#include "ISvm.h"

namespace phd { namespace svm
{
KernelTypes stringToKernelType(gsl::cstring_span<> kernelName);

gsl::cstring_span<> kernelTypeToString(KernelTypes kernelType);

std::string svmTypeToString(SvmTypes svmType);
}}// namespace phd::svm
