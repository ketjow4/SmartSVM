
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/TraningSetCrossover.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2012 Kawulok and Nalepa - Support vector machines training data selection using a genetic algorithm
class GaSvmCrossover : public TrainingSetCrossover
{
public:
    using chromosomeType = SvmTrainingSetChromosome;

    explicit GaSvmCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                            unsigned int numberOfClasses);

    chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB) override;
};
} // namespace svmComponents
