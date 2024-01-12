
#include <unordered_set>
#include "CrossoverCompensation.h"
#include "SvmComponentsExceptions.h"

namespace svmComponents
{
CrossoverCompensation::CrossoverCompensation(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                             std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                             unsigned int numberOfClasses)
    : m_trainingSet(trainingSet)
    , m_rngEngine(std::move(rngEngine))
    , m_numberOfClasses(numberOfClasses)
{
}



void CrossoverCompensation::compensate(geneticComponents::Population<SvmTrainingSetChromosome>& population,
                                       const std::vector<unsigned int>& compensationInfo)
{
    if (compensationInfo.size() == population.size())
    {
        auto index = 0u;
        const auto targets = m_trainingSet.getLabels();
        const auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));

        for (auto& individual : population)
        {
            std::vector<unsigned int> classCount(m_numberOfClasses, static_cast<const unsigned>(individual.getDataset().size() / m_numberOfClasses));
            auto numberOfClassExamples = (individual.getDataset().size() + compensationInfo[index]) / m_numberOfClasses;
            auto dataset = individual.getDataset();
            auto trainingSet = individual.convertToSet();

            int j = 0;
            while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies)
            {
                return classIndicies != numberOfClassExamples;
            }))
            {
                auto randomValue = m_rngEngine->getRandom(trainingSetID);
                if (classCount[static_cast<int>(targets[randomValue])] < numberOfClassExamples && // less that desired number of class examples
                    trainingSet.emplace(static_cast<int>(randomValue)).second) // is unique
                {
                    dataset.emplace_back(DatasetVector(randomValue, static_cast<std::uint8_t>(targets[randomValue])));
                    ++classCount[static_cast<int>(targets[randomValue])];
                }

                j++;
                if (j > 10000) //FIX THIS IN FUTURE
                    break;
            }
            individual.updateDataset(dataset);
            index++;

        }
    }
    else
    {
        throw ContainersSizeInequality("libSvmComponents CrossoverCompensation::compensate", compensationInfo.size(), population.size());
    }
}
} // namespace svmComponents
