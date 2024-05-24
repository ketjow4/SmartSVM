#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticFeaturesSuperIndividualsGeneration
{
public:
    MemeticFeaturesSuperIndividualsGeneration(std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator,
                                              unsigned int numberOfClasses);

    geneticComponents::Population<SvmFeatureSetMemeticChromosome> createPopulation(uint32_t populationSize,
                                                                                   const std::vector<Feature>& supportVectorPool,
                                                                                   unsigned int numberOfClassExamples);

private:
    std::vector<SvmFeatureSetMemeticChromosome> generate(unsigned int populationSize,
                                                         const std::vector<Feature>& supportVectorPool,
                                                         unsigned int numberOfClassExamples);

    const std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    const unsigned int m_numberOfClasses;
};

inline MemeticFeaturesSuperIndividualsGeneration::MemeticFeaturesSuperIndividualsGeneration(
    std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator, unsigned numberOfClasses)
    : m_rngEngine(std::move(randomNumberGenerator))
    , m_numberOfClasses(numberOfClasses)
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

inline geneticComponents::Population<SvmFeatureSetMemeticChromosome> MemeticFeaturesSuperIndividualsGeneration::createPopulation(uint32_t populationSize,
                                                                                                                                 const std::vector<Feature>&
                                                                                                                                 supportVectorPool,
                                                                                                                                 unsigned numberOfClassExamples)
{
    if (populationSize == 0)
    {
        throw geneticComponents::PopulationIsEmptyException();
    }
    if (supportVectorPool.empty())
    {
        throw EmptySupportVectorPool();
    }

    auto minLabelCount = static_cast<unsigned int>(supportVectorPool.size());

    if (numberOfClassExamples > minLabelCount)
    {
        numberOfClassExamples = minLabelCount;
    }

    auto population = generate(populationSize, supportVectorPool, numberOfClassExamples);

    return geneticComponents::Population<SvmFeatureSetMemeticChromosome>(std::move(population));
}

inline bool emplacementCondition(const Feature& drawnSupportVector,
                          std::unordered_set<std::uint64_t>& trainingSet)
{
    return trainingSet.emplace(static_cast<int>(drawnSupportVector.id)).second; // is unique
}

inline std::vector<SvmFeatureSetMemeticChromosome> MemeticFeaturesSuperIndividualsGeneration::generate(unsigned populationSize,
                                                                                                       const std::vector<Feature>& supportVectorPool,
                                                                                                       unsigned numberOfClassExamples)
{
    std::vector<SvmFeatureSetMemeticChromosome> population(populationSize);
    auto featuresSetID = std::uniform_int_distribution<int>(0, static_cast<int>(supportVectorPool.size() - 1));

    std::generate(population.begin(), population.end(), [&]
    {
        std::unordered_set<std::uint64_t> trainingSet;
        std::vector<Feature> superIndividualDataset;
        superIndividualDataset.reserve(numberOfClassExamples);
        unsigned int featuresCount = 0;
        while (featuresCount != numberOfClassExamples)
        {
            auto randomValue = m_rngEngine->getRandom(featuresSetID);
            const auto& drawnSupportVector = supportVectorPool[randomValue];
            if (emplacementCondition(drawnSupportVector, trainingSet))
            {
                superIndividualDataset.emplace_back(Feature(drawnSupportVector));
                featuresCount++;
            }
        }
        return SvmFeatureSetMemeticChromosome(std::move(superIndividualDataset));
    });
    return population;
}
} // namespace svmComponents
