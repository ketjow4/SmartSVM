
#pragma once

#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/IOperator.h"
#include "LibGeneticComponents/Population.h"

namespace geneticComponents
{
template<typename chromosomeType>
class IMutationOperator : public IOperator<chromosomeType>
{
    static_assert(std::is_base_of<BaseChromosome, chromosomeType>::value, "Cannot do mutation for class not derived from BaseChromosome");
public:
    virtual ~IMutationOperator() = default;

    virtual void mutatePopulation(Population<chromosomeType>& population) = 0;
    virtual void mutateChromosome(chromosomeType& chromosome) = 0;
    
    void operator()(Population<chromosomeType>& population) override;
};

#pragma warning( push )
#pragma warning( disable : 4505) // @wdudzik Warning C4505 unreferenced local function has been removed, this is referenced by other libraries

template <typename chromosomeType>
void IMutationOperator<chromosomeType>::operator()(Population<chromosomeType>& population)
{
    mutatePopulation(population);
}

#pragma warning( pop ) 
} // namespace geneticComponents
