
#include "SvmKernelChromosome.h"

namespace svmComponents
{
SvmKernelChromosome::SvmKernelChromosome() : m_kernelType(phd::svm::KernelTypes::Custom)
{
}

SvmKernelChromosome::SvmKernelChromosome(phd::svm::KernelTypes kernelType, const std::vector<double>& parameters, bool isRegression)
    : m_kernelType(kernelType)
    , m_kernelParameters(parameters)
    , m_isRegression(isRegression)
{
}

bool SvmKernelChromosome::operator==(const SvmKernelChromosome& right) const
{
    constexpr auto epsilon = 1e-10;
    auto numberOfParameters = m_kernelParameters.size();
    auto notUniqueParameters = 0u;
    for (auto k = 0u; k < numberOfParameters; k++)
    {
        if (fabs(m_kernelParameters[k] - right.getKernelParameters()[k]) < epsilon)
        {
            notUniqueParameters++;
        }
    }
    return notUniqueParameters == numberOfParameters;
}
} // namespace svmComponents