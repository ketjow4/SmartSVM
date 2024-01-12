
#pragma once

#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/Population.h"

namespace geneticComponents
{
template<typename chromosomeType>
class IPopulationGeneration
{
    static_assert(std::is_base_of<BaseChromosome, chromosomeType>::value, "Cannot create population for class not derived from BaseChromosome");
public:
    virtual ~IPopulationGeneration() = default;

    virtual Population<chromosomeType> createPopulation(uint32_t populationSize) = 0;
};
} // namespace geneticComponents
