
#pragma once

#include <string>

#include "SvmLib/ISvm.h"
#include "libPlatform/PlatformException.h"

namespace svmComponents
{
class UnsupportedKernelTypeException final : public platform::PlatformException
{
public:
    explicit UnsupportedKernelTypeException(const std::string& kernelName);

    explicit UnsupportedKernelTypeException(phd::svm::KernelTypes kernelType);
};

class GridSearchUnsupportedKernelTypeException final : public platform::PlatformException
{
public:
    explicit GridSearchUnsupportedKernelTypeException(phd::svm::KernelTypes kernelType);
};

class UnsupportedGridException final : public platform::PlatformException
{
public:
    explicit UnsupportedGridException(const std::string& gridName);
};

enum class DatasetType
{
    Training,
    Validation,
    Test,
    ValidationOrTest
};

class EmptyDatasetException final : public platform::PlatformException
{
public:
    explicit EmptyDatasetException(DatasetType datasetType);

private:
    static std::string translateEnum(DatasetType datasetType);
};

class UntrainedSvmClassifierException final : public platform::PlatformException
{
public:
    explicit UntrainedSvmClassifierException()
        : PlatformException{"Untrained svm classifier"}
    {
    }
};

class UnsupportedImageFormat final : public platform::PlatformException
{
public:
    explicit UnsupportedImageFormat()
        : PlatformException{"Unsupported image format"}
    {
    }
};

class ValueOfClassExamplesIsTooHighForDataset final : public platform::PlatformException
{
public:
    explicit ValueOfClassExamplesIsTooHighForDataset(unsigned int numberOfClassExamples);
};

class UnknownEnumType final : public platform::PlatformException
{
public:
    explicit UnknownEnumType(const std::string& name, const std::string& enumName);
    explicit UnknownEnumType(const std::string& enumName);
};

class CrossoverParentsSizeInequality final : public platform::PlatformException
{
public:
    explicit CrossoverParentsSizeInequality(std::size_t parentASize, std::size_t parentBSize);
};

class ContainersSizeInequality final : public platform::PlatformException
{
public:
    explicit ContainersSizeInequality(const std::string& where, std::size_t containerASize, std::size_t containerBSize);
};

class EmptySupportVectorPool final : public platform::PlatformException
{
public:
    explicit EmptySupportVectorPool()
        : PlatformException{"Empty support vector pool"}
    {
    }
};

class RandomNumberGeneratorNullPointer final : public platform::PlatformException
{
public:
    explicit RandomNumberGeneratorNullPointer()
        : PlatformException{"Null pointer in rng"}
    {
    }
};

class MemberNullPointer final : public platform::PlatformException
{
public:
    explicit MemberNullPointer(const std::string& nameOfMember);
};

class TooSmallNumberOfClasses final : public platform::PlatformException
{
public:
    explicit TooSmallNumberOfClasses(unsigned int numberOfClasses);
};

class TooSmallPopulationSize final : public platform::PlatformException
{
public:
    explicit TooSmallPopulationSize(unsigned int actualValue, unsigned int minimum);
};

class ValueNotInRange final : public platform::PlatformException
{
    
public:
    template <typename arthmeticType>
    explicit ValueNotInRange(const std::string& valueName, arthmeticType value, arthmeticType min, arthmeticType max);
};

template <typename arthmeticType>
ValueNotInRange::ValueNotInRange(const std::string& valueName, arthmeticType value, arthmeticType min, arthmeticType max)
    : PlatformException("Value: " + valueName + " is: " + std::to_string(value) + " where it should be in range ("
        + std::to_string(min) + ", " + std::to_string(max) + ")")
{
    static_assert(std::is_arithmetic<arthmeticType>::value, "ArthmeticType must be integer or floating point number");
}

class CannotClassifyWithOptmialThreshold final : public platform::PlatformException
{
public:
    explicit CannotClassifyWithOptmialThreshold();
};

class OneClassValidationSet final : public platform::PlatformException
{
public:
    explicit OneClassValidationSet();
};

} // namespace svmComponents