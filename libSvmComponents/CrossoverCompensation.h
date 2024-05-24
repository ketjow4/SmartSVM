
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/Population.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class CrossoverCompensation
{
public:
    CrossoverCompensation(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                          std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                          unsigned int numberOfClasses);

    void compensate(geneticComponents::Population<SvmTrainingSetChromosome>& population,
                    const std::vector<unsigned int>& compensationInfo);

private:
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClasses;
};
} // namespace svmComponents
