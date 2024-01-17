#pragma once

#include "libSvmComponents/SuperIndividualsCreation.h"

namespace svmStrategies
{
class SuperIndividualCreationStrategy
{
public:
    explicit SuperIndividualCreationStrategy(svmComponents::SuperIndividualsCreation& generationAlgorithm);

    std::string getDescription() const;
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> launch(unsigned int populationSize,
                                                                                  std::vector<svmComponents::DatasetVector>& supportVectorPool,
                                                                                  unsigned int numberOfClassExamples);

private:
    svmComponents::SuperIndividualsCreation& m_generationAlgorithm;
};
} // namespace svmStrategies
