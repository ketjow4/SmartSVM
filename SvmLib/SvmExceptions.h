
#pragma once

#include "ISvm.h"

namespace phd { namespace svm
{
class UnsupportedKernelTypeException final : public platform::PlatformException
{
public:
    explicit UnsupportedKernelTypeException(const std::string& kernelName);

    explicit UnsupportedKernelTypeException(KernelTypes kernelType);
};

class UnknownSvmTypeException final : public platform::PlatformException
{
public:
    explicit UnknownSvmTypeException(SvmTypes svmType);
};

class EmptyTraningDataSet final : public platform::PlatformException
{
public:
    explicit EmptyTraningDataSet();
};

class ValueNotPositiveException final : public platform::PlatformException
{
public:
    explicit ValueNotPositiveException(const std::string& valueName);
};

class ValueNotInRange final : public platform::PlatformException
{

public:
    template <typename arthmeticType>
    explicit ValueNotInRange(const std::string& valueName, arthmeticType value, arthmeticType min, arthmeticType max);
};

class UntrainedSvmClassifierException final : public platform::PlatformException
{
public:
    explicit UntrainedSvmClassifierException();
};

template <typename arthmeticType>
ValueNotInRange::ValueNotInRange(const std::string& valueName, arthmeticType value, arthmeticType min, arthmeticType max)
    : PlatformException("Value: " + valueName + " is: " + std::to_string(value) + " where it should be in range ("
        + std::to_string(min) + ", " + std::to_string(max) + ")")
{
    static_assert(std::is_arithmetic<arthmeticType>::value, "ArthmeticType must be integer or floating point number");
}

class UnknownEnumType final : public platform::PlatformException
{
public:
    explicit UnknownEnumType(const std::string& name, const std::string& enumName);
    explicit UnknownEnumType(const std::string& enumName);
};
}}// namespace phd::svm
