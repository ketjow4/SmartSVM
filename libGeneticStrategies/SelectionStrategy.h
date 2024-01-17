#pragma once

#include "libGeneticComponents/ISelectionOperator.h"
#include "libGeneticComponents/GeneticExceptions.h"

namespace geneticStrategies
{
template <class chromosome>
class SelectionStrategy
{
public:
    explicit SelectionStrategy(geneticComponents::ISelectionOperator<chromosome>& selectionMethod);

    std::string getDescription() const;
    auto launch(geneticComponents::Population<chromosome>& population,
                geneticComponents::Population<chromosome>& other);

    geneticComponents::ISelectionOperator<chromosome>& getMethod()
    {
        return m_selectionMethod;
    }
	
private:
    geneticComponents::ISelectionOperator<chromosome>& m_selectionMethod;
};

template <class chromosome>
SelectionStrategy<chromosome>::SelectionStrategy(geneticComponents::ISelectionOperator<chromosome>& selectionMethod)
    : m_selectionMethod(selectionMethod)
{
}

template <class chromosome>
std::string SelectionStrategy<chromosome>::getDescription() const
{
    return "Select individuals for next generation from 2 provided population using given method";
}

template <class chromosome>
auto SelectionStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population,
                                           geneticComponents::Population<chromosome>& other)
{
    std::vector<chromosome> combined;
    combined.reserve(population.size() + other.size());
    combined.insert(combined.end(), population.begin(), population.end());
    combined.insert(combined.end(), other.begin(), other.end());
    geneticComponents::Population<chromosome> nextGeneration = geneticComponents::Population<chromosome>(combined);

    nextGeneration.applyOperator(m_selectionMethod);
    return nextGeneration;
}
} // namespace svmStrategies
