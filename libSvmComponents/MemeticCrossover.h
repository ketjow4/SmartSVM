
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/TraningSetCrossover.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticCrossover : public TrainingSetCrossover
{
public:
    explicit MemeticCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                              unsigned int numberOfClasses);

    SvmTrainingSetChromosome crossoverChromosomes(const SvmTrainingSetChromosome& parentA, const SvmTrainingSetChromosome& parentB) override;
};
} // namespace svmComponents
