
#pragma once

#include <memory>
#include <random>
#include "LibGeneticComponents/Population.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/BinaryChromosome.h"
#include "LibGeneticComponents/IMutationOperator.h"
#include "libPlatform/Percent.h"

namespace geneticComponents
{
template <class binaryChromosome>
class BitFlipMutation : public IMutationOperator<binaryChromosome>
{
    static_assert(std::is_base_of<BinaryChromosome, binaryChromosome>::value, "Cannot do binary mutation for class not derived from BinaryChromosome");
public:
    explicit BitFlipMutation(platform::Percent flipProbability,
                             std::unique_ptr<random::IRandomNumberGenerator> rngEngine);

    void mutatePopulation(Population<binaryChromosome>& population) override;
    void mutateChromosome(binaryChromosome& chromosome) override;

private:
    platform::Percent m_flipProbability;
    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
};

template <class binaryChromosome>
BitFlipMutation<binaryChromosome>::BitFlipMutation(platform::Percent flipProbability,
                                                   std::unique_ptr<random::IRandomNumberGenerator> rngEngine)
    : m_flipProbability(flipProbability)
    , m_rngEngine(std::move(rngEngine))
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

template <class binaryChromosome>
void BitFlipMutation<binaryChromosome>::mutatePopulation(Population<binaryChromosome>& population)
{
    if (population.empty())
    {
        throw PopulationIsEmptyException();
    }

    for (auto& chromosome : population)
    {
        mutateChromosome(chromosome);
    }
}

template <class binaryChromosome>
void BitFlipMutation<binaryChromosome>::mutateChromosome(binaryChromosome& chromosome)
{
    std::bernoulli_distribution flipProbability(m_flipProbability.m_percentValue);
    auto genes = chromosome.getGenes();

    std::for_each(genes.begin(), genes.end(), [this,&flipProbability](auto gene)
          {
              if (m_rngEngine->getRandom(flipProbability))
              {
                  gene = !gene;
              }
          });
    chromosome.updateGenes(genes);
}
} // namespace geneticComponents
