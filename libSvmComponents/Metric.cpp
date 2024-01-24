

#include "Metric.h"

namespace svmComponents
{
Metric::Metric()
    : m_fitness(0)
    , m_additionalValue(0)
    , m_confusionMatrix()
{
}

Metric::Metric(double fitness)
    : m_fitness(fitness)
	, m_additionalValue(0)
    , m_confusionMatrix()
{
}

Metric::Metric(double fitness, ConfusionMatrix matrix)
    : m_fitness(fitness)
	, m_additionalValue(0)
    , m_confusionMatrix(std::move(matrix))
{
}

Metric::Metric(double fitness, double additionalValue, ConfusionMatrix matrix)
    : m_fitness(fitness)
    , m_additionalValue(additionalValue)
    , m_confusionMatrix(std::move(matrix))
{
}
} // namespace svmComponents
