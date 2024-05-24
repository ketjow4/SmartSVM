
#pragma once
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/ICrossoverSelection.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/SvmUtils.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class CompensationInformation
{
public:
    explicit CompensationInformation(std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator,
                                     unsigned int numberOfClasses);

    std::vector<unsigned int> generate(const std::vector<geneticComponents::Parents<SvmTrainingSetChromosome>>& parents,
                                       unsigned int numberOfClassExamples) const;

private:
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    const unsigned int m_numberOfClasses;
};
} // namespace svmComponents
