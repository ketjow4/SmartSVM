
#pragma once

#include "LibGeneticComponents/Population.h"

namespace geneticComponents
{
class BinaryChromosome : public virtual BaseChromosome
{
public:
    virtual ~BinaryChromosome() = default;

    virtual const std::vector<bool>& getGenes() const = 0;
    virtual void updateGenes(const std::vector<bool>&) = 0;
};
} // namespace geneticComponents
