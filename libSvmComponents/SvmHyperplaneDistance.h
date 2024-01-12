#pragma once

#include "libSvmComponents/ISvmMetricsCalculator.h"

namespace svmComponents
{
enum MetricMode
{
	defaultOption,
	zeroOut,
	nonlinearDecrease,
	boundryCheck
};

struct SvmAnswer;

class SvmHyperplaneDistance : public ISvmMetricsCalculator
{
public:
	explicit SvmHyperplaneDistance(bool useDistance)
		: m_useDistance(useDistance)
		, m_multicriteriaOptimization(false)
		, m_DistanceBestPositive(0)
		, m_DistanceBestNegative(0)
		, m_distanceSet(false)
		, m_mode(defaultOption)
		, m_useBias(false)
		, m_useSingleClassPrediction(false)
	{}

	explicit SvmHyperplaneDistance(bool useDistance, bool useBias, bool useSingleClassPrediction)
		: m_useDistance(useDistance)
		, m_multicriteriaOptimization(false)
		, m_DistanceBestPositive(0)
		, m_DistanceBestNegative(0)
		, m_distanceSet(false)
		, m_mode(defaultOption)
		, m_useBias(useBias)
		, m_useSingleClassPrediction(useSingleClassPrediction)
	{}

	explicit SvmHyperplaneDistance(bool useDistance, bool multicriteriaOptimization)
		: m_useDistance(useDistance)
		, m_multicriteriaOptimization(multicriteriaOptimization)
		, m_DistanceBestPositive(0)
		, m_DistanceBestNegative(0)
		, m_distanceSet(false)
		, m_mode(defaultOption)
		, m_useBias(false)
		, m_useSingleClassPrediction(false)
	{}


	//explicit SvmHyperplaneDistance(bool useDistance, double distanceBestPositive, double distanceBestNegative, MetricMode mode)
	//	: m_useDistance(useDistance)
	//	, m_multicriteriaOptimization(false)
	//	, m_DistanceBestPositive(distanceBestPositive)
	//	, m_DistanceBestNegative(distanceBestNegative)
	//	, m_distanceSet(true)
	//	, m_mode(mode)
	//	, m_useBias(false)
	//	, m_useSingleClassPrediction(false)
	//{}

	explicit SvmHyperplaneDistance(bool useDistance, double distanceBestPositive, double distanceBestNegative, MetricMode mode, bool useBias, bool useSingleClassPrediction)
		: m_useDistance(useDistance)
		, m_multicriteriaOptimization(false)
		, m_DistanceBestPositive(distanceBestPositive)
		, m_DistanceBestNegative(distanceBestNegative)
		, m_distanceSet(true)
		, m_mode(mode)
		, m_useBias(useBias)
		, m_useSingleClassPrediction(useSingleClassPrediction)
	{}
	
	Metric calculateMetric(const BaseSvmChromosome& individual,
	                       const dataset::Dataset<std::vector<float>, float>& testSamples) const override;

	
	Metric calculateMetric(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool isTestSet) const override;


	void calculateThresholds(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples);

private:
	void classifySet(std::shared_ptr<phd::svm::ISvm> svmModel,
	                 gsl::span<const float> targets,
	                 gsl::span<const std::vector<float>> samples,
	                 std::vector<SvmAnswer>& results) const;
	void setThresholds(std::shared_ptr<phd::svm::ISvm> svmModel,
	                   double& max_distance_raw,
	                   double& min_distance_raw,
	                   double& max_distance_normalized,
	                   double& min_distance_normalized) const;

	bool m_useDistance;
	bool m_multicriteriaOptimization;

	bool m_distanceSet;
	bool m_useBias;
	bool m_useSingleClassPrediction;
	
	double m_DistanceBestPositive;
	double m_DistanceBestNegative;
	MetricMode m_mode;
};
} // namespace svmComponents
