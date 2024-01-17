#include "libDataset/Dataset.h"
#include "libSvmComponents/SvmTraining.h"
#include "libSvmComponents/GridSearchCrossValidation.h"
#include "GridSearchCrossValidationStrategy.h"

namespace svmStrategies
{
GridSearchCrossValidationStrategy::GridSearchCrossValidationStrategy(svmComponents::GridSearchConfiguration& algorithmConfig)
    : m_gridSearch(algorithmConfig)
{
}

std::string GridSearchCrossValidationStrategy::getDescription() const
{
    return "Do cross validation on traning set for grid search and return model trained with best parameters";
}

geneticComponents::Population<svmComponents::SvmKernelChromosome> GridSearchCrossValidationStrategy::launch(
    geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
    const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
    return m_gridSearch.run(population, trainingDataset);
}
} // namespace svmStrategies
