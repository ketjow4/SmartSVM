#include "CrossoverCompensationStrategy.h"
#include "libGeneticComponents/Population.h"

namespace svmStrategies
{
CrossoverCompensationStrategy::CrossoverCompensationStrategy(svmComponents::CrossoverCompensation& crossoverCompensationAlgorithm)
    : m_crossoverCompensation(crossoverCompensationAlgorithm)
{
}

std::string CrossoverCompensationStrategy::getDescription() const
{
    return "Compensate individuals in population with elements from traning set";
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> CrossoverCompensationStrategy::launch(
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
    std::vector<unsigned int> compensationInfo)
{
    m_crossoverCompensation.compensate(population, compensationInfo);
    return population;
}
} // namespace svmStrategies
