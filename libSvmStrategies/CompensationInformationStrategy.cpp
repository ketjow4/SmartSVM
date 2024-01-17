#include "CompensationInformationStrategy.h"

namespace svmStrategies
{
CompensationInformationStrategy::CompensationInformationStrategy(
    svmComponents::CompensationInformation& compensationGenerationAlgorithm)
    : m_compensationGeneration(compensationGenerationAlgorithm)
{
}

std::string CompensationInformationStrategy::getDescription() const
{
    return "This element generates information for compensation strategy based on parents and number of class examples";
}

std::vector<unsigned int> CompensationInformationStrategy::launch(
    const std::vector<geneticComponents::Parents<svmComponents::SvmTrainingSetChromosome>>& parents,
    unsigned int numberOfClassExamples)
{
    return m_compensationGeneration.generate(parents, numberOfClassExamples);
}
} // namespace svmStrategies
