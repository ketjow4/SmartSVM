
#pragma once

#include "LibGeneticComponents/IChromosomeCache.h"

namespace geneticStrategies
{
template <class chromosome>
class AddToBinaryCacheStrategy 
{
public:
    explicit AddToBinaryCacheStrategy(geneticComponents::IChromosomeCache<chromosome>& cache);

  
    std::string getDescription() const;
    void launch(geneticComponents::Population<chromosome>& population);

private:
    geneticComponents::IChromosomeCache<chromosome>& m_cache;
};

template <class chromosome>
AddToBinaryCacheStrategy<chromosome>::AddToBinaryCacheStrategy(geneticComponents::IChromosomeCache<chromosome>& cache)
    :m_cache(cache)
{
}

template <class chromosome>
std::string AddToBinaryCacheStrategy<chromosome>::getDescription() const
{
    return  "Adds elements from population to cache so calculated values can be used later.";
}

template <class chromosome>
void AddToBinaryCacheStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population)
{
        for(const auto& individual : population)
        {
            m_cache.addToCache(individual);
        }
}
} // namespace svmStrategies
