

#include "SvmMetricFactory.h"
#include "SvmComponentsExceptions.h"
#include "SvmAccuracyMetric.h"
#include "SvmAucMetric.h"
#include "R2_regression.h"
#include "SvmAucprcMetric.h"
#include "SvmHyperplaneDistance.h"

namespace svmComponents
{
std::unique_ptr<ISvmMetricsCalculator> SvmMetricFactory::create(svmMetricType metricType)
{
    switch (metricType)
    {
    case svmMetricType::Accuracy:
        return std::make_unique<SvmAccuracyMetric>();
    case svmMetricType::Auc:
        return std::make_unique<SvmAucMetric>();
    case svmMetricType::R2:
        return std::make_unique<R2_regression>();
    case svmMetricType::PrAuc:
        return std::make_unique<SvmAucprcMetric>();
    case svmMetricType::BalancedAccuracy:
        return std::make_unique<SvmBalancedAccuracyMetric>();
    case svmMetricType::HyperplaneDistance:
        return std::make_unique<SvmHyperplaneDistance>(true); //useDistance = true
    case svmMetricType::MCC:
        return std::make_unique<SvmMCCMetric>();
    case svmMetricType::CertainAccuracy:
        return std::make_unique<SvmHyperplaneDistance>(false); //useDistance = false
    default:
        throw UnknownEnumType(typeid(svmMetricType).name());
    }
}
} // namespace svmComponents
