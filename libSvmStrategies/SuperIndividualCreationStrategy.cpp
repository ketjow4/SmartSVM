#include "SuperIndividualCreationStrategy.h"

namespace svmStrategies
{
SuperIndividualCreationStrategy::SuperIndividualCreationStrategy(svmComponents::SuperIndividualsCreation& generationAlgorithm)
    : m_generationAlgorithm(generationAlgorithm)
{
}

std::string SuperIndividualCreationStrategy::getDescription() const
{
    return "Generates individuals composed of support vectors only";
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> SuperIndividualCreationStrategy::launch(unsigned int populationSize,
                                                                                                               std::vector<svmComponents::DatasetVector>&
                                                                                                               supportVectorPool,
                                                                                                               unsigned int numberOfClassExamples)
{
    auto newPopulation = m_generationAlgorithm.createPopulation(populationSize, supportVectorPool, numberOfClassExamples);

    return newPopulation;
}
} // namespace svmStrategies
