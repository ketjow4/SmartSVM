

#pragma once
#include "ConfusionMatrix.h"
#include <boost/optional/optional.hpp>

namespace svmComponents
{
class Metric
{
public:
    Metric();
	
    explicit Metric(double fitness);
    Metric(double fitness, ConfusionMatrix matrix);

    Metric(double fitness, double additionalValue, ConfusionMatrix matrix);

    double m_fitness;
    double m_additionalValue;
    boost::optional<ConfusionMatrix> m_confusionMatrix;
};
} // namespace svmComponents
