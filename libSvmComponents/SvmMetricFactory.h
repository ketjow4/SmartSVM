#pragma once

#include <memory>
#include "libSvmComponents/ISvmMetricsCalculator.h"

namespace svmComponents
{
enum class svmMetricType
{
    Unknown,
    Accuracy,
    Auc,
    R2,
	PrAuc,
	BalancedAccuracy,
	HyperplaneDistance,
	MCC,
	CertainAccuracy
};

class SvmMetricFactory 
{
public:
    static std::unique_ptr<ISvmMetricsCalculator> create(svmMetricType metricType);
};
} // namespace svmComponents