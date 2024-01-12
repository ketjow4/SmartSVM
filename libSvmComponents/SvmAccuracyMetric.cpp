

#include "libDataset/Dataset.h"
#include "SvmAccuracyMetric.h"
#include "BaseSvmChromosome.h"
#include "ConfusionMatrixMetrics.h"

namespace svmComponents
{
Metric SvmAccuracyMetric::calculateMetric(const BaseSvmChromosome& individual,
                                          const dataset::Dataset<std::vector<float>, float>& testSamples) const
{
    ConfusionMatrix matrix(individual, testSamples);
    return Metric(Accuracy(matrix), matrix);
	
    /*ConfusionMatrixMulticlass matrix(individual, testSamples);
    return Metric(Accuracy(matrix), ConfusionMatrix(0,0,0,0));*/
}

Metric SvmAccuracyMetric::calculateMetric(const BaseSvmChromosome& individual,
                                          const dataset::Dataset<std::vector<float>, float>& testSamples,
                                          bool /*isTestSet*/) const
{
	return calculateMetric(individual, testSamples);
}

Metric SvmAccuracyMetric::calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool /*isTestSet*/,
	bool parrarell) const
{
    ConfusionMatrix matrix(individual, testSamples, parrarell);
    return Metric(Accuracy(matrix), std::move(matrix));
}

Metric SvmMCCMetric::calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples) const
{
    ConfusionMatrix matrix(individual, testSamples);
    return Metric(MCC(matrix), std::move(matrix));
}

Metric SvmMCCMetric::calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool /*isTestSet*/) const
{
    return calculateMetric(individual, testSamples);
}

Metric SvmBalancedAccuracyMetric::calculateMetric(const BaseSvmChromosome& individual,
                                                  const dataset::Dataset<std::vector<float>, float>& testSamples) const
{
	ConfusionMatrix matrix(individual, testSamples);
	return Metric(BalancedAccuracy(matrix), std::move(matrix));

	/*ConfusionMatrixMulticlass matrix(individual, testSamples);
	return Metric(BalancedAccuracy(matrix), ConfusionMatrix(0, 0, 0, 0));*/
}
} // namespace svmComponents
