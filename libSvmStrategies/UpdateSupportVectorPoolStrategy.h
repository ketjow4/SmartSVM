#pragma once

#include "libSvmComponents/SupportVectorPool.h"

namespace svmStrategies
{
class UpdateSupportVectorPoolStrategy
{
public:
    explicit UpdateSupportVectorPoolStrategy(svmComponents::SupportVectorPool& updateMethod);

    std::string getDescription() const;
    const std::vector<svmComponents::DatasetVector>& launch(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
                                                            const dataset::Dataset<std::vector<float>, float>& trainingSet);

private:
    svmComponents::SupportVectorPool& m_updateMethod;
};
} // namespace svmStrategies
