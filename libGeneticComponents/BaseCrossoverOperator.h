
#pragma once

#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/Population.h"
#include "LibGeneticComponents/ICrossoverSelection.h"

namespace geneticComponents
{
template<typename chromosomeType>
class BaseCrossoverOperator 
{
    static_assert(std::is_base_of<BaseChromosome, chromosomeType>::value, "Cannot do crossover for class not derived from BaseChromosome");
public:
    virtual ~BaseCrossoverOperator() = default;

    virtual Population<chromosomeType> crossoverParents(const std::vector<Parents<chromosomeType>>& parents);
    virtual chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB) = 0;
    
};

#pragma warning( push )
#pragma warning( disable : 4505) // @wdudzik Warning C4505 unreferenced local function has been removed, this is referenced by other libraries

template <typename chromosomeType>
Population<chromosomeType> BaseCrossoverOperator<chromosomeType>::crossoverParents(const std::vector<Parents<chromosomeType>>& parents)
{
    std::vector<chromosomeType> children(parents.size());

    std::transform(parents.begin(), parents.end(), children.begin(), [this](const auto& parentsPair)
    {
        return  this->crossoverChromosomes(parentsPair.first, parentsPair.second);
    });

    return geneticComponents::Population<chromosomeType>(std::move(children));
}

#pragma warning( pop ) 
} // namespace geneticComponents
