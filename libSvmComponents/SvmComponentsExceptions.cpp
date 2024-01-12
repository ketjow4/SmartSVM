
#include <string>
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "SvmLib/EnumsTranslations.h"

namespace svmComponents
{
UnsupportedKernelTypeException::UnsupportedKernelTypeException(const std::string& kernelName)
    : PlatformException("Kernel: " + kernelName + " is unsupported")
{
}

UnsupportedKernelTypeException::UnsupportedKernelTypeException(phd::svm::KernelTypes kernelType)
    : PlatformException("Kernel: " + gsl::to_string(kernelTypeToString(kernelType)) + " is unsupported")
{
}

GridSearchUnsupportedKernelTypeException::GridSearchUnsupportedKernelTypeException(phd::svm::KernelTypes kernelType)
    : PlatformException("Kernel: " + gsl::to_string(kernelTypeToString(kernelType)) + " is unsupported in grid search algorithm")
{
}

UnsupportedGridException::UnsupportedGridException(const std::string& gridName)
    : PlatformException("Grid: " + gridName + " is not supported")
{
}

EmptyDatasetException::EmptyDatasetException(DatasetType datasetType)
    : PlatformException(translateEnum(datasetType) + " dataset cannot be empty")
{
}

std::string EmptyDatasetException::translateEnum(DatasetType datasetType)
{
    switch (datasetType)
    {
    case DatasetType::Training:
        return "Training";
    case DatasetType::Validation:
        return "Validation";
    case DatasetType::Test:
        return "Test";
    case DatasetType::ValidationOrTest:
        return "ValidationOrTest";
    default:
        throw UnknownEnumType("", typeid(DatasetType).name());;
    }
}

ValueOfClassExamplesIsTooHighForDataset::ValueOfClassExamplesIsTooHighForDataset(unsigned int numberOfClassExamples)
    : PlatformException("Number of class examples: " + std::to_string(numberOfClassExamples) + " is too high for selected dataset")
{
}

UnknownEnumType::UnknownEnumType(const std::string& name, const std::string& enumName)
    : PlatformException("Enum type:" + name + " is unknown name in:" + enumName)
{
}

UnknownEnumType::UnknownEnumType(const std::string& enumName)
    : PlatformException("Unknown enum value for: " + enumName)
{
}

CrossoverParentsSizeInequality::CrossoverParentsSizeInequality(std::size_t parentASize, std::size_t parentBSize)
    : PlatformException("Parents sizes differ in crossover operator. ParentA: " + std::to_string(parentASize) + " ParentB: " + std::to_string(parentBSize))
{
}

ContainersSizeInequality::ContainersSizeInequality(const std::string& where, std::size_t containerASize, std::size_t containerBSize)
    : PlatformException("Containers in: " + where + " differ in size. ContainerA: " + std::to_string(containerASize) +
                        " ContainerB : " + std::to_string(containerBSize))
{
}

MemberNullPointer::MemberNullPointer(const std::string& nameOfMember): PlatformException("Member named: " + nameOfMember + " was nullptr when passed to constructor")
{
}

TooSmallNumberOfClasses::TooSmallNumberOfClasses(unsigned numberOfClasses)
    : PlatformException("Number of classes should be at least 2. Current is:" + std::to_string(numberOfClasses))
{
}

TooSmallPopulationSize::TooSmallPopulationSize(unsigned int actualValue, unsigned int minimum)
    : PlatformException("Size of population should be at least " + std::to_string(minimum) + ". Current is:" + std::to_string(actualValue))
{
}

CannotClassifyWithOptmialThreshold::CannotClassifyWithOptmialThreshold()
    : PlatformException("Svm model cannot classify with optmial threshold. Use AUC metric on trening to enable that.")
{
}

OneClassValidationSet::OneClassValidationSet()
    : PlatformException("Validations set contains only one class")
{
}
} // namespace svmComponents
