#pragma once

#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libGeneticComponents/Population.h"

namespace geneticStrategies
{
template <class chromosome>
class CrossoverStrategy
{
public:
    explicit CrossoverStrategy(geneticComponents::BaseCrossoverOperator<chromosome>& crossoverAlgorithm);

    std::string getDescription() const;
    auto launch(std::vector<geneticComponents::Parents<chromosome>>& parents);

private:
    geneticComponents::BaseCrossoverOperator<chromosome>& m_crossoverAlgorithm;
};

template <class chromosome>
CrossoverStrategy<chromosome>::CrossoverStrategy(geneticComponents::BaseCrossoverOperator<chromosome>& crossoverAlgorithm)
    : m_crossoverAlgorithm(crossoverAlgorithm)
{
}

template <class chromosome>
std::string CrossoverStrategy<chromosome>::getDescription() const
{
    return "Generic strategy to do crossover with given method and type of chromosome";
}

template <class chromosome>
auto CrossoverStrategy<chromosome>::launch(std::vector<geneticComponents::Parents<chromosome>>& parents)
{
    return m_crossoverAlgorithm.crossoverParents(parents);
}
} // namespace svmStrategies
