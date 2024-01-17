#pragma once

#include "libSvmComponents/SvmConfigStructures.h"
#include "libSvmComponents/GridSearchCrossValidation.h"

namespace svmStrategies
{
class GridSearchCrossValidationStrategy
{
public:
    explicit GridSearchCrossValidationStrategy(svmComponents::GridSearchConfiguration& algorithmConfig);

    std::string getDescription() const;
    geneticComponents::Population<svmComponents::SvmKernelChromosome> launch(
        geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
        const dataset::Dataset<std::vector<float>, float>& trainingDataset);

private:
    svmComponents::GridSearchCrossValidation m_gridSearch;
};
} // namespace svmStrategies
