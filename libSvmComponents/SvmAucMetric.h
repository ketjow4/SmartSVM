

#pragma once

#include "libSvmComponents/ISvmMetricsCalculator.h"

namespace svmComponents
{
struct AucResults;

class SvmAucMetric : public ISvmMetricsCalculator
{
public:
    SvmAucMetric(bool parallel = false)
        : m_parallel(parallel)
	{}

    SvmAucMetric(std::string metric): m_metric(metric)
	{}

	
    /** @wdudzik Confusion matrix that is calculated in AUC is based on the point of ROC with highest accuracy
     *  This point (threshold for probability output) is also set to SVM model so it can be used later.
     */
    Metric calculateMetric(const BaseSvmChromosome& individual,
                           const dataset::Dataset<std::vector<float>, float>& testSamples) const override;

    Metric calculateMetric(const BaseSvmChromosome& individual,
                           const dataset::Dataset<std::vector<float>, float>& testSamples,
                           bool isTestSet) const override;

private:
    double trapezoidArea(double x1, double x2, double y1, double y2) const;

    /*Based on article An introduction to ROC analysis
    Tom Fawcett
    Institute for the Study of Learning and Expertise, 2164 Staunton Court, Palo Alto, CA 94306, USA*/
    AucResults auc(std::vector<std::pair<double, int>>& probabilityTargetPair, int negativeCount, int positiveCount) const;

    Metric handleGroups(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const;


    std::string m_metric;
    static constexpr int m_positiveClassValue = 1;
    bool m_parallel;
};
} // namespace svmComponents
