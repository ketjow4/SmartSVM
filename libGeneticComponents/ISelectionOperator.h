
#pragma once

#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/Population.h"

namespace geneticComponents
{
template<typename chromosomeType>
class ISelectionOperator : public IOperator<chromosomeType>
{
    static_assert(std::is_base_of<BaseChromosome, chromosomeType>::value, "Cannot do selection for class not derived from BaseChromosome");
public: 
    virtual ~ISelectionOperator() = default;

    virtual Population<chromosomeType> selectNextGeneration(Population<chromosomeType>& currentGeneration) = 0;
    
    void operator()(Population<chromosomeType>& population) override;

};

#pragma warning( push )
#pragma warning( disable : 4505) // @wdudzik Warning C4505 unreferenced local function has been removed, this is referenced by other libraries

template <typename chromosomeType>
void ISelectionOperator<chromosomeType>::operator()(Population<chromosomeType>& population)
{
    auto nextGeneration = selectNextGeneration(population);
    population.swap(nextGeneration);
}

#pragma warning( pop ) 
} // namespace geneticComponents
