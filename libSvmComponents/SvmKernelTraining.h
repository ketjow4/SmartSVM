
#pragma once

#include "libSvmComponents/ISvmTraining.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libSvmComponents/SvmConfigStructures.h"

namespace svmComponents
{
class SvmKernelTraining : public ISvmTraining<SvmKernelChromosome>
{
public:
    explicit SvmKernelTraining(const SvmAlgorithmConfiguration& svmConfig, bool probabilityNeeded = false);

    void trainPopulation(geneticComponents::Population<SvmKernelChromosome>& population,
                         const dataset::Dataset<std::vector<float>, float>& trainingData) override;

private:
    const SvmAlgorithmConfiguration m_svmConfig;
    const bool m_probabilityNeeded;
    std::exception_ptr m_lastException;
};
} // namespace svmComponents
