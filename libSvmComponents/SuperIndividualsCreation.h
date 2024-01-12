
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class SuperIndividualsCreation
{
public:
    SuperIndividualsCreation(std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
                               unsigned int numberOfClasses);

    geneticComponents::Population<SvmTrainingSetChromosome> createPopulation(uint32_t populationSize,
                                                                             const std::vector<DatasetVector>& supportVectorPool,
                                                                             unsigned int numberOfClassExamples);

private:
    std::vector<SvmTrainingSetChromosome> generate(unsigned int populationSize,
                                                   const std::vector<DatasetVector>& supportVectorPool,
                                                   unsigned int numberOfClassExamples);

    const std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    const unsigned int m_numberOfClasses;
};
} // namespace svmComponents
