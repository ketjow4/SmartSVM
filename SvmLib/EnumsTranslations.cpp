
#include <map>
#include "EnumsTranslations.h"
#include "SvmExceptions.h"

namespace phd { namespace svm
{
KernelTypes stringToKernelType(std::string& kernelName)
{
    const static std::map<std::string, KernelTypes> translations =
    {
        {"RBF", KernelTypes::Rbf},
        {"POLY", KernelTypes::Poly},
        {"SIGMOID", KernelTypes::Sigmoid},
        {"CHI2", KernelTypes::Chi2},
        {"LINEAR", KernelTypes::Linear},
		{"RBF_POLY_GLOBAL", KernelTypes::RBF_POLY_GLOBAL}
    };

    auto iterator = translations.find(kernelName);
    if (iterator != translations.end())
    {
        return iterator->second;
    }
    throw UnsupportedKernelTypeException(kernelName);
}

std::string kernelTypeToString(KernelTypes kernelType)
{
    switch (kernelType)
    {
    case KernelTypes::Custom:
        return std::string("Custom kernel");
    case KernelTypes::Linear:
        return std::string("Linear kernel");
    case KernelTypes::Poly:
        return std::string("Polynomial kernel");
    case KernelTypes::Rbf:
        return std::string("Radial basis function (RBF)");
    case KernelTypes::Sigmoid:
        return std::string("Sigmoid kernel");
    case KernelTypes::Chi2:
        return std::string("Exponential Chi2 kernel");
    case KernelTypes::Inter:
        return std::string("Histogram intersection kernel");
    default:
        return std::string("Unknown kernel type");
    }
}

std::string svmTypeToString(SvmTypes svmType)
{
    switch (svmType)
    {
    case SvmTypes::CSvc: return "CSvc";
    case SvmTypes::NuSvc: return "NuSvc";
    case SvmTypes::OneClass: return "OneClass";
    case SvmTypes::EpsSvr: return "EpsSvr";
    case SvmTypes::NuSvr: return "NuSvr";
    default:
        return "Unknown svm type";
    }
}
}}// namespace phd::svm
