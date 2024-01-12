
#pragma once

#include <unordered_map>
#include <random>
#include "LibGeneticComponents/BinaryChromosome.h"
#include "LibGeneticComponents/IChromosomeCache.h"

namespace geneticComponents
{
template <typename binaryChromosome>
class BinaryChromosomeCache : public IChromosomeCache<binaryChromosome>
{
    static_assert(std::is_base_of<BinaryChromosome, binaryChromosome>::value, "Cannot use BinaryChromosomeCache for chromosome not derived from BinaryChromosome");
public:
    BinaryChromosomeCache() = default;

    void addToCache(const binaryChromosome& chromosomeWithFitness) override;

    void clearCache() override;

    bool searchInCacheAndUpdate(binaryChromosome& chromosome) override;

private:
    std::unordered_map<std::vector<bool>, binaryChromosome> m_cache;
};

// @maksym : The three functions below generate false C4505 warning, despite being used in tests and strategies
#pragma warning( push )
#pragma warning( disable : 4505)
template <typename binaryChromosome>
void BinaryChromosomeCache<binaryChromosome>::addToCache(const binaryChromosome& chromosomeWithFitness)
{    
    m_cache[chromosomeWithFitness.getGenes()] = chromosomeWithFitness;
}

template <typename binaryChromosome>
void BinaryChromosomeCache<binaryChromosome>::clearCache()
{
    m_cache.clear();
}

template <typename binaryChromosome>
bool BinaryChromosomeCache<binaryChromosome>::searchInCacheAndUpdate(binaryChromosome& chromosome)
{
    auto cachedElement = m_cache.find(chromosome.getGenes());
    if(cachedElement != m_cache.end())
    {
        chromosome = (*cachedElement).second;
        return true;
    }
    return false;
}
#pragma warning( pop ) 
} // namespace geneticComponents
