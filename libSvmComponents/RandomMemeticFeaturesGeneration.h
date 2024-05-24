#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/Population.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
class RandomMemeticFeaturesGeneration : public geneticComponents::IPopulationGeneration<SvmFeatureSetMemeticChromosome>
{
public:
    explicit RandomMemeticFeaturesGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                             std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                             unsigned int numberOfClassExamples)
        : m_trainingSet(trainingSet)
        , m_rngEngine(std::move(rngEngine))
        , m_numberOfClassExamples(numberOfClassExamples)
    {
        if (m_numberOfClassExamples > trainingSet.getSample(0).size())
        {
            throw ValueOfClassExamplesIsTooHighForDataset(m_numberOfClassExamples);
        }
    }

    geneticComponents::Population<SvmFeatureSetMemeticChromosome> createPopulation(uint32_t populationSize) override
    {
        if (populationSize == 0)
        {
            throw geneticComponents::PopulationIsEmptyException();
        }

        auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.getSample(0).size() - 1));
        std::vector<SvmFeatureSetMemeticChromosome> population(populationSize);

        std::generate(population.begin(), population.end(), [&]
                      {
                          std::unordered_set<std::uint64_t> trainingSet;
                          std::vector<Feature> chromosomeDataset;
                          chromosomeDataset.reserve(m_numberOfClassExamples);
                          std::vector<unsigned int> classCount(1, 0);
                          while (std::any_of(classCount.begin(), classCount.end(), [this](const auto& classIndicies) { return classIndicies != m_numberOfClassExamples;  }))
                          {
                              auto randomValue = m_rngEngine->getRandom(trainingSetID);
                              if (classCount[0] < m_numberOfClassExamples &&    // less that desired number of class examples
                                  trainingSet.emplace(static_cast<int>(randomValue)).second)       // is unique
                              {
                                  chromosomeDataset.emplace_back(Feature(randomValue));
                                  classCount[0]++;
                              }
                          }
                          return SvmFeatureSetMemeticChromosome(std::move(chromosomeDataset));
                      });
        return geneticComponents::Population<SvmFeatureSetMemeticChromosome>(std::move(population));
    }

private:
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClassExamples;
};
} // namespace svmComponents
