#pragma once
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/ICrossoverSelection.h"
#include "libSvmComponents/SvmUtils.h"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticFeaturesCompensationGeneration
{
public:
    explicit MemeticFeaturesCompensationGeneration(std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator);

    std::vector<unsigned int> generate(const std::vector<geneticComponents::Parents<SvmFeatureSetMemeticChromosome>>& parents,
                                       unsigned int numberOfClassExamples) const;

private:
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
};
} // namespace svmComponents
