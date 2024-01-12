

#pragma once

#include "libSvmComponents/ISvmMetricsCalculator.h"

namespace svmComponents
{
class SvmAccuracyMetric : public ISvmMetricsCalculator
{
public:
    Metric calculateMetric(const BaseSvmChromosome& individual,
                           const dataset::Dataset<std::vector<float>, float>& testSamples) const override;
    Metric calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const override;

    Metric calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet, bool parrarell) const;
};


class SvmMCCMetric : public ISvmMetricsCalculator
{
public:
    Metric calculateMetric(const BaseSvmChromosome& individual,
        const dataset::Dataset<std::vector<float>, float>& testSamples) const override;
    Metric calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const override;
};


class SvmBalancedAccuracyMetric : public ISvmMetricsCalculator
{
public:
    Metric calculateMetric(const BaseSvmChromosome& individual,
        const dataset::Dataset<std::vector<float>, float>& testSamples) const override;

    Metric calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool /*isTestSet*/) const override
    {
        return calculateMetric(individual, testSamples);
    }
};
} // namespace svmComponents
