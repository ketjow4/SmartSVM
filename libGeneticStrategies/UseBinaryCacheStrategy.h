#pragma once

#include "LibGeneticComponents/IChromosomeCache.h"

namespace geneticStrategies
{
template <class chromosome>
class UseToBinaryCacheStrategy
{
public:
    explicit UseToBinaryCacheStrategy(geneticComponents::IChromosomeCache<chromosome>& cache);

    std::string getDescription() const;
    auto launch(geneticComponents::Population<chromosome>& population);

private:
    geneticComponents::IChromosomeCache<chromosome>& m_cache;
};

template <class chromosome>
UseToBinaryCacheStrategy<chromosome>::UseToBinaryCacheStrategy(geneticComponents::IChromosomeCache<chromosome>& cache)
    : m_cache(cache)
{
}

template <class chromosome>
std::string UseToBinaryCacheStrategy<chromosome>::getDescription() const
{
    return "Use values stored in cache to divide population into 2 gropus (with known and unknown fitness).";
}

template <class chromosome>
auto UseToBinaryCacheStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population)
{
        std::vector<chromosome> unknownFitnessPopulation;
        std::vector<chromosome> alreadyKnownFitness;
        unknownFitnessPopulation.reserve(population.size());
        alreadyKnownFitness.reserve(population.size());

        for (auto& individual : population)
        {
            if (m_cache.searchInCacheAndUpdate(individual))
            {
                alreadyKnownFitness.emplace_back(individual);
            }
            else
            {
                unknownFitnessPopulation.emplace_back(individual);
            }
        }

        return std::make_tuple(geneticComponents::Population<chromosome>(unknownFitnessPopulation),
                               geneticComponents::Population<chromosome>(alreadyKnownFitness));
}
} // namespace svmStrategies
