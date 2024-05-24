#pragma once

#include "libSvmComponents/ISvmMetricsCalculator.h"

namespace svmComponents
{
    class R2_regression : public ISvmMetricsCalculator
    {
    public:
        Metric calculateMetric(const BaseSvmChromosome& individual,
                               const dataset::Dataset<std::vector<float>, float>& testSamples) const override;

    	Metric calculateMetric(const BaseSvmChromosome& /*individual*/, 
            const dataset::Dataset<std::vector<float>, float>& /*testSamples*/,
	        bool /*isTestSet*/) const override
    	{
            throw std::runtime_error("Not implemented R2 regression ");
    	}
    };
} // namespace svmComponents
