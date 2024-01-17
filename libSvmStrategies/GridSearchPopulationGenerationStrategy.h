#pragma once

#include "libSvmComponents/BaseKernelGridSearch.h"

namespace svmStrategies
{
class GridSearchPopulationGenerationStrategy
{
public:
    explicit GridSearchPopulationGenerationStrategy(svmComponents::BaseKernelGridSearch& gridSearchKernel);

    std::string getDescription() const;
    geneticComponents::Population<svmComponents::SvmKernelChromosome> launch(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population);

private:
    svmComponents::BaseKernelGridSearch& m_gridSearchKernel;
};
} // namespace svmStrategies
