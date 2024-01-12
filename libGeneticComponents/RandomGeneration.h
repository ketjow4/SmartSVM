
#pragma once

#include <memory>
#include "LibGeneticComponents/Population.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/BinaryChromosome.h"
#include "LibGeneticComponents/IPopulationGeneration.h"
#include "libPlatform/Percent.h"

namespace geneticComponents
{
template <class binaryChromosome>
class RandomGeneration : public IPopulationGeneration<binaryChromosome>
{
    static_assert(std::is_base_of<BinaryChromosome, binaryChromosome>::value, "Cannot do binary crossover for class not derived from BinaryChromosome");

public:
    RandomGeneration(unsigned int chromosomeSize,
                     platform::Percent percentageOfFill,
                     std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator);
    Population<binaryChromosome> createPopulation(uint32_t populationSize) override;

protected:
    virtual binaryChromosome createIndividual();

    unsigned int m_chromosomeSize;
    platform::Percent m_percentageOfFill;
    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;

    static constexpr auto m_miniumChromosomeSize = 2u;
};

template <class binaryChromosome>
RandomGeneration<binaryChromosome>::RandomGeneration(unsigned int chromosomeSize,
                                                     platform::Percent percentageOfFill,
                                                     std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator)
    : m_chromosomeSize(chromosomeSize)
    , m_percentageOfFill(percentageOfFill)
    , m_rngEngine(std::move(randomNumberGenerator))
{
    if (m_chromosomeSize < m_miniumChromosomeSize)
    {
        throw TooSmallChromosomeSize(m_chromosomeSize, m_miniumChromosomeSize);
    }
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

template <class binaryChromosome>
Population<binaryChromosome> RandomGeneration<binaryChromosome>::createPopulation(uint32_t populationSize)
{
    if (populationSize == 0)
    {
        throw PopulationIsEmptyException();
    }

    std::vector<binaryChromosome> population(populationSize);

    std::generate(population.begin(), population.end(), [this]
              {
                  return createIndividual();
              });
    return Population<binaryChromosome>(population);
}

template <class binaryChromosome>
binaryChromosome RandomGeneration<binaryChromosome>::createIndividual()
{
    std::vector<bool> genes(m_chromosomeSize);
    std::bernoulli_distribution fullfillment(m_percentageOfFill.m_percentValue);

    std::generate(genes.begin(), genes.end(), [this, &fullfillment]
              {
                  return m_rngEngine->getRandom(fullfillment);
              });
    binaryChromosome individual(std::move(genes));
    return individual;
}
} // namespace geneticComponents
