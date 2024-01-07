
#include <map>
#include "EnumsTranslations.h"
#include "SvmExceptions.h"

namespace phd { namespace svm
{
struct StringSpanComparator final
{
    typedef StringSpanComparator is_transparent;

    bool operator()(const gsl::cstring_span<>& lhs, const gsl::cstring_span<>& rhs) const
    {
        return lhs < rhs;
    }
};

KernelTypes stringToKernelType(gsl::cstring_span<> kernelName)
{
    const static std::map<std::string, KernelTypes, StringSpanComparator> translations =
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
    throw UnsupportedKernelTypeException(gsl::to_string(kernelName));
}

gsl::cstring_span<> kernelTypeToString(KernelTypes kernelType)
{
    switch (kernelType)
    {
    case KernelTypes::Custom:
        return gsl::ensure_z("Custom kernel");
    case KernelTypes::Linear:
        return gsl::ensure_z("Linear kernel");
    case KernelTypes::Poly:
        return gsl::ensure_z("Polynomial kernel");
    case KernelTypes::Rbf:
        return gsl::ensure_z("Radial basis function (RBF)");
    case KernelTypes::Sigmoid:
        return gsl::ensure_z("Sigmoid kernel");
    case KernelTypes::Chi2:
        return gsl::ensure_z("Exponential Chi2 kernel");
    case KernelTypes::Inter:
        return gsl::ensure_z("Histogram intersection kernel");
    default:
        return gsl::ensure_z("Unknown kernel type");
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
