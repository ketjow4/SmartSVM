#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/Population.h"
#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticFeatureCompensation
{
public:
    MemeticFeatureCompensation(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                               std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                               unsigned int numberOfClasses);

    void compensate(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                    const std::vector<unsigned int>& compensationInfo);

private:
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClasses;
};


} // namespace svmComponents
