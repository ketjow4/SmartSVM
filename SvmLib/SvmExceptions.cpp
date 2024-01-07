
#include "SvmExceptions.h"
#include "EnumsTranslations.h"

namespace phd { namespace svm
{
UnsupportedKernelTypeException::UnsupportedKernelTypeException(const std::string& kernelName)
    : PlatformException("Kernel: " + kernelName + " is unsupported")
{
}

UnsupportedKernelTypeException::UnsupportedKernelTypeException(KernelTypes kernelType)
    : PlatformException("Kernel: " + gsl::to_string(kernelTypeToString(kernelType)) + " is unsupported")
{
}

UnknownSvmTypeException::UnknownSvmTypeException(SvmTypes svmType)
    : PlatformException("Svm type: " + svmTypeToString(svmType) + "is unknown to current impelmentation")
{
}

EmptyTraningDataSet::EmptyTraningDataSet()
    : PlatformException("Dataset passed to svm was empty")
{
}

ValueNotPositiveException::ValueNotPositiveException(const std::string& valueName)
    : PlatformException("Parameter " + valueName + " have to be positive")
{
}

UntrainedSvmClassifierException::UntrainedSvmClassifierException()
    : PlatformException("Untrained svm classifier")
{
}

UnknownEnumType::UnknownEnumType(const std::string& name, const std::string& enumName)
    : PlatformException("Enum type:" + name + " has unknown name of:" + enumName)
{
}

UnknownEnumType::UnknownEnumType(const std::string& enumName)
    : PlatformException("Unknown value for " + enumName)
{
}
}}// namespace phd::svm
