#include "MemeticFeatureCompensation.h"
#include "SmallerPoolExperiment.h"

namespace svmComponents
{
MemeticFeatureCompensation::MemeticFeatureCompensation(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                       std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine, unsigned numberOfClasses)
    : m_trainingSet(trainingSet)
    , m_rngEngine(std::move(rngEngine))
    , m_numberOfClasses(numberOfClasses)
{
}

void MemeticFeatureCompensation::compensate(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                                            const std::vector<unsigned>& compensationInfo)
{
    if (compensationInfo.size() == population.size())
    {
        auto index = 0u;
        auto featureSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.getSample(0).size() - 1));

        for (auto& individual : population)
        {
            auto featureCount = static_cast<unsigned int>(individual.getDataset().size());
            const auto numberOfFeatures = (individual.getDataset().size() + compensationInfo[index]);
            auto featureVector = individual.getDataset();
            auto featureSet = individual.convertToSet();
            int tries = 0;
            while (featureCount != numberOfFeatures && tries < 200)
            {
                const auto randomValue = m_rngEngine->getRandom(featureSetID);
                if (SmallerPool::instance().isOk(randomValue) && featureSet.emplace(static_cast<int>(randomValue)).second) // is unique
                {
                    featureVector.emplace_back(Feature(randomValue));
                    featureCount++;
                }
                tries++;
            }
            individual.updateDataset(featureVector);
            index++;
        }
    }
    else
    {
        throw ContainersSizeInequality("libSvmComponents MemeticFeatureCompensation::compensate", compensationInfo.size(), population.size());
    }
}
} // namespace svmComponents
