
#pragma once
#include "SvmKernelTraining.h"

namespace svmComponents
{
class GridSearchCrossValidation
{
public:
    explicit GridSearchCrossValidation(GridSearchConfiguration& algorithmConfig);
   

    geneticComponents::Population<SvmKernelChromosome> run(geneticComponents::Population<SvmKernelChromosome>& population,
                                                           const dataset::Dataset<std::vector<float>, float>& trainingDataset);

private:
    void trainClassifiers(geneticComponents::Population<SvmKernelChromosome>& population,
                          const dataset::Dataset<std::vector<float>, float>& trainingDataset,
                          std::vector<double>& results);

    void averageResults(std::vector<double>& results, geneticComponents::Population<SvmKernelChromosome>& population);

    SvmKernelChromosome findBestParameters(const geneticComponents::Population<SvmKernelChromosome>& population,
                                           const std::vector<double>& results);

    void validate(geneticComponents::Population<SvmKernelChromosome>& population,
                  std::vector<double>& results,
                  const dataset::Dataset<std::vector<float>, float>& validationSet);

    GridSearchConfiguration& m_algorithmConfig;
    SvmKernelTraining m_trainingMethod;
};
} // namespace svmComponents
