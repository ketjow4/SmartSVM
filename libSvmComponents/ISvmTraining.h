
#pragma once

#include <libDataset/Dataset.h>
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/BaseSvmChromosome.h"

namespace svmComponents
{
template<typename chromosomeType>
class ISvmTraining
{
    static_assert(std::is_base_of<BaseSvmChromosome, chromosomeType>::value, "Cannot do training for class not derived from BaseSvmChromosome");

public:
    virtual ~ISvmTraining() = default;

    virtual void trainPopulation(geneticComponents::Population<chromosomeType>& population,
                                 const dataset::Dataset<std::vector<float>, float>& trainingData) = 0;
};
} // namespace svmComponents
