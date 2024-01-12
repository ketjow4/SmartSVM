
#pragma once
#include "libSvmComponents/BaseSvmChromosome.h"

namespace svmComponents
{
class SvmKernelChromosome : public BaseSvmChromosome
{
public:
    explicit SvmKernelChromosome();

    explicit SvmKernelChromosome(phd::svm::KernelTypes kernelType, const std::vector<double>& parameters, bool isRegression);

    phd::svm::KernelTypes getKernelType() const;

    const std::vector<double>& getKernelParameters() const;
    
    double operator[](size_t index) const;

    void updateKernelParameters(std::vector<double>& parameters);

    bool operator==(const SvmKernelChromosome& right) const;

    bool isRegression() const
    {
        return m_isRegression;
    }

private:
    phd::svm::KernelTypes m_kernelType;
    std::vector<double> m_kernelParameters;
    bool m_isRegression;
};

inline phd::svm::KernelTypes SvmKernelChromosome::getKernelType() const
{
    return m_kernelType;
}

inline const std::vector<double>& SvmKernelChromosome::getKernelParameters() const
{
    return  m_kernelParameters;
}

inline double SvmKernelChromosome::operator[](size_t index) const
{
    return m_kernelParameters[index];
}

inline void SvmKernelChromosome::updateKernelParameters(std::vector<double>& parameters)
{
    m_kernelParameters.swap(parameters);
}
} // namespace svmComponents