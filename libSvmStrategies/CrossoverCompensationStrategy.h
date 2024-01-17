#pragma once

#include "libSvmComponents/CrossoverCompensation.h"

namespace svmStrategies
{
class CrossoverCompensationStrategy
{
public:
    explicit CrossoverCompensationStrategy(svmComponents::CrossoverCompensation& crossoverCompensationAlgorithm);

    std::string getDescription() const ;
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> launch(
        geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
        std::vector<unsigned int> compensationInfo);

private:
    svmComponents::CrossoverCompensation& m_crossoverCompensation;
};
} // namespace svmStrategies
