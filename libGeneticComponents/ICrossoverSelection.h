
#pragma once

#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/Population.h"

namespace geneticComponents
{
template<class chromosome>
using Parents = std::pair<chromosome, chromosome>;

using Indexes = std::pair<int, int>;

template<typename chromosomeType>
class ICrossoverSelection
{
    static_assert(std::is_base_of<BaseChromosome, chromosomeType>::value, "Cannot do selection for class not derived from BaseChromosome");
public:
    virtual ~ICrossoverSelection() = default;

    virtual Parents<chromosomeType> chooseParents(Population<chromosomeType>& population) = 0;

    virtual Indexes chooseIndexes(Population<chromosomeType>& population) = 0;
};
} // namespace geneticComponents
