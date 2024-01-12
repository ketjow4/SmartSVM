
#pragma once

namespace geneticComponents
{
template <typename Chromosome>
class IChromosomeCache
{
public:
    virtual ~IChromosomeCache() = default;

    virtual void addToCache(const Chromosome& chromosomeWithFitness) = 0;
    virtual void clearCache() = 0;
    virtual bool searchInCacheAndUpdate(Chromosome& chromosome) = 0;
};
} // namespace geneticComponents