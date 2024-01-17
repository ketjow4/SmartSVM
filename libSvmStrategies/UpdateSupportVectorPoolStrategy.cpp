#include "UpdateSupportVectorPoolStrategy.h"
#include "libSvmComponents/SvmComponentsExceptions.h"

namespace svmStrategies
{
UpdateSupportVectorPoolStrategy::UpdateSupportVectorPoolStrategy(svmComponents::SupportVectorPool& updateMethod)
    : m_updateMethod(updateMethod)
{
}

std::string UpdateSupportVectorPoolStrategy::getDescription() const
{
    return "Updates support vector pool with all support vectors from population";
}

const std::vector<svmComponents::DatasetVector>& UpdateSupportVectorPoolStrategy::launch(
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
    const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    m_updateMethod.updateSupportVectorPool(population, trainingSet);

    return m_updateMethod.getSupportVectorPool();
}
} // namespace svmStrategies
