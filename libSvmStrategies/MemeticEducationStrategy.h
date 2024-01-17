#pragma once

#include "libSvmComponents/EducationOfTrainingSet.h"

namespace svmStrategies
{
class MemeticEducationStrategy
{
public:
    explicit MemeticEducationStrategy(svmComponents::EducationOfTrainingSet& educationAlgorithm);

    std::string getDescription() const;
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& launch(
        geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
        std::vector<svmComponents::DatasetVector>& supportVectorPool,
        std::vector<geneticComponents::Parents<svmComponents::SvmTrainingSetChromosome>>& parents,
        const dataset::Dataset<std::vector<float>, float>& trainingSet);

private:
    svmComponents::EducationOfTrainingSet& m_educationOfTrainingSet;
};
} // namespace svmStrategies
