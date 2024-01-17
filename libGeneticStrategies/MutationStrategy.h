#pragma once

#include "libGeneticComponents/IMutationOperator.h"
#include "libGeneticComponents/Population.h"
#include "libGeneticComponents/GeneticExceptions.h"

namespace geneticStrategies
{
template <class chromosome>
class MutationStrategy
{
public:
    explicit MutationStrategy(geneticComponents::IMutationOperator<chromosome>& mutationAlgorithm);

    std::string getDescription() const;
    auto launch(geneticComponents::Population<chromosome>& population);

private:
    geneticComponents::IMutationOperator<chromosome>& m_mutationAlgorithm;
};

template <class chromosome>
MutationStrategy<chromosome>::MutationStrategy(geneticComponents::IMutationOperator<chromosome>& mutationAlgorithm)
    : m_mutationAlgorithm(mutationAlgorithm)
{
}

template <class chromosome>
std::string MutationStrategy<chromosome>::getDescription() const
{
    return "Mutate the population with given method, changes population that is given on input";
}

template <class chromosome>
auto MutationStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population)
{
    population.applyOperator(m_mutationAlgorithm);
    return population;
}
} // namespace svmStrategies
