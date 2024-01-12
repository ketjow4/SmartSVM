
#include <unordered_set>
#include "SuperIndividualsCreation.h"
#include "SvmUtils.h"

namespace svmComponents
{
using namespace geneticComponents;

SuperIndividualsCreation::SuperIndividualsCreation(std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
                                                   unsigned int numberOfClasses)
    : m_rngEngine(std::move(randomNumberGenerator))
    , m_numberOfClasses(numberOfClasses)
{
    if(m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

bool emplacementCondition(const DatasetVector& drawnSupportVector,
                          const std::vector<unsigned int>& classCount,
                          const unsigned int numberOfClassExamples,
                          std::unordered_set<std::uint64_t>& trainingSet)
{
    return classCount[static_cast<int>(drawnSupportVector.classValue)] < numberOfClassExamples && // less that desired number of class examples
            trainingSet.emplace(static_cast<int>(drawnSupportVector.id)).second; // is unique
}

std::vector<SvmTrainingSetChromosome> SuperIndividualsCreation::generate(unsigned int populationSize,
                                                                         const std::vector<DatasetVector>& supportVectorPool,
                                                                         unsigned int numberOfClassExamples)
{
    std::vector<SvmTrainingSetChromosome> population(populationSize);
    auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(supportVectorPool.size() - 1));

    std::generate(population.begin(), population.end(), [&]
          {
              std::unordered_set<std::uint64_t> trainingSet;
              std::vector<DatasetVector> superIndividualDataset;
              superIndividualDataset.reserve(numberOfClassExamples * m_numberOfClasses);
              std::vector<unsigned int> classCount(m_numberOfClasses, 0);
              while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies)
                         {
                             return classIndicies != numberOfClassExamples;
                         }))
              {
                  auto randomValue = m_rngEngine->getRandom(trainingSetID);
                  const auto& drawnSupportVector = supportVectorPool[randomValue];
                  if (emplacementCondition(drawnSupportVector, classCount, numberOfClassExamples, trainingSet))
                  {
                      superIndividualDataset.emplace_back(DatasetVector(drawnSupportVector));
                      classCount[static_cast<int>(drawnSupportVector.classValue)]++;
                  }
              }
              return SvmTrainingSetChromosome(std::move(superIndividualDataset));
          });
    return population;
}

Population<SvmTrainingSetChromosome> SuperIndividualsCreation::createPopulation(uint32_t populationSize,
                                                                                const std::vector<DatasetVector>& supportVectorPool,
                                                                                unsigned int numberOfClassExamples)
{
    if (populationSize == 0)
    {
        throw PopulationIsEmptyException();
    }
    if (supportVectorPool.empty())
    {
        throw EmptySupportVectorPool();
    }

    auto labelsCount = svmUtils::countLabels(m_numberOfClasses, supportVectorPool);
    auto minLabelCount = std::min_element(labelsCount.begin(), labelsCount.end());

    if (numberOfClassExamples > *minLabelCount)
    {
        numberOfClassExamples = *minLabelCount;
    }

    auto population = generate(populationSize, supportVectorPool, numberOfClassExamples);

    return Population<SvmTrainingSetChromosome>(std::move(population));
}
} // namespace svmComponents
