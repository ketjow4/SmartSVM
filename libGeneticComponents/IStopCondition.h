
#pragma once

#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/Population.h"

namespace geneticComponents
{
template<typename chromosomeType>
class IStopCondition
{
    static_assert(std::is_base_of<BaseChromosome, chromosomeType>::value, "Cannot check for stop condition for class not derived from BaseChromosome");
public:
    virtual ~IStopCondition() = default;
    
    virtual bool isFinished(const Population<chromosomeType>& population) = 0;

    virtual void reset() = 0;
};
} // namespace geneticComponents
