#include "MemeticEducationStrategy.h"

namespace svmStrategies
{
MemeticEducationStrategy::MemeticEducationStrategy(svmComponents::EducationOfTrainingSet& educationAlgorithm)
    : m_educationOfTrainingSet(educationAlgorithm)
{
}

std::string MemeticEducationStrategy::getDescription() const
{
    return "Modifies current population with education algorithm";
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& MemeticEducationStrategy::launch(
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
    std::vector<svmComponents::DatasetVector>& supportVectorPool,
    std::vector<geneticComponents::Parents<svmComponents::SvmTrainingSetChromosome>>& parents,
    const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    m_educationOfTrainingSet.educatePopulation(population, supportVectorPool, parents, trainingSet);

    return population;
}
} // namespace svmStrategies
