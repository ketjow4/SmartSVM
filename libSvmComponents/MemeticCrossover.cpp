
#include "MemeticCrossover.h"

namespace svmComponents
{
MemeticCrossover::MemeticCrossover(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                   unsigned int numberOfClasses)
    : TrainingSetCrossover(std::move(rngEngine), numberOfClasses)
{
}

SvmTrainingSetChromosome MemeticCrossover::crossoverChromosomes(const SvmTrainingSetChromosome& parentA, const SvmTrainingSetChromosome& parentB)
{
    auto sizeOfChild = std::max(parentA.getDataset().size(), parentB.getDataset().size());
    auto datasetSize = sizeOfChild;
    auto classExamples = sizeOfChild / m_numberOfClasses;

    auto child = crossoverInternals(parentA, parentB, static_cast<unsigned int>(datasetSize), static_cast<unsigned int>(classExamples));

    return SvmTrainingSetChromosome(std::move(child));
}
} // namespace svmComponents
