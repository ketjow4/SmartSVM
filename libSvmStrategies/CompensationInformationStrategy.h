#pragma once

#include "libSvmComponents/CompensationInformation.h"

namespace svmStrategies
{
class CompensationInformationStrategy
{
public:
    explicit CompensationInformationStrategy(svmComponents::CompensationInformation& compensationGenerationAlgorithm);

    std::string getDescription() const;
    std::vector<unsigned int> launch(
        const std::vector<geneticComponents::Parents<svmComponents::SvmTrainingSetChromosome>>& parents,
        unsigned int numberOfClassExamples);

private:
    svmComponents::CompensationInformation& m_compensationGeneration;
};
} // namespace svmStrategies
