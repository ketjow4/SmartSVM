

#pragma once

#include <opencv2/ml.hpp>
#include "libDataset/Dataset.h"
#include "libSvmComponents/Metric.h"

namespace svmComponents
{
class BaseSvmChromosome;

class ISvmMetricsCalculator
{
public:
    virtual ~ISvmMetricsCalculator() = default;

    virtual Metric calculateMetric(const BaseSvmChromosome& individual,
                                   const dataset::Dataset<std::vector<float>, float>& testSamples) const = 0;

    virtual Metric calculateMetric(const BaseSvmChromosome& individual,
                                   const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const = 0;
};
} // namespace svmComponents
