

#pragma once
#include "ConfusionMatrix.h"
#include <optional>

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
    std::optional<ConfusionMatrix> m_confusionMatrix;
};
} // namespace svmComponents
