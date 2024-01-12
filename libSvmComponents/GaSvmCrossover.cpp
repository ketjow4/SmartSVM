
#include "GaSvmCrossover.h"
#include "SvmComponentsExceptions.h"

namespace svmComponents
{
GaSvmCrossover::GaSvmCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                               unsigned int numberOfClasses)
    : TrainingSetCrossover(std::move(rngEngine), numberOfClasses)
{
    if(numberOfClasses < 2)
    {
        throw TooSmallNumberOfClasses(numberOfClasses);
    }
}

GaSvmCrossover::chromosomeType GaSvmCrossover::crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB)
{
    if(parentA.getDataset().size() != parentB.getDataset().size())
    {
        throw CrossoverParentsSizeInequality(parentA.getDataset().size(), parentB.getDataset().size());
    }

    auto datasetSize = static_cast<unsigned int>(parentA.getDataset().size());
    auto classExamples = static_cast<unsigned int>(datasetSize / m_numberOfClasses);

    auto child = crossoverInternals(parentA, parentB, datasetSize, classExamples);
    return SvmTrainingSetChromosome(std::move(child));
}
} // namespace svmComponents
