#pragma once

#include "libGeneticComponents/GeneticExceptions.h"
#include "libGeneticComponents/ICrossoverSelection.h"

namespace geneticStrategies
{
template <class chromosome>
class CrossoverParentSelectionStrategy
{
public:
    explicit CrossoverParentSelectionStrategy(geneticComponents::ICrossoverSelection<chromosome>& crossoverSelection);

    std::string getDescription() const;
    auto launch(geneticComponents::Population<chromosome>& population);

private:
    geneticComponents::ICrossoverSelection<chromosome>& m_crossoverSelection;
};

template <class chromosome>
CrossoverParentSelectionStrategy<chromosome>::CrossoverParentSelectionStrategy(geneticComponents::ICrossoverSelection<chromosome>& crossoverSelection)
    : m_crossoverSelection(crossoverSelection)
{
}

template <class chromosome>
std::string CrossoverParentSelectionStrategy<chromosome>::getDescription() const
{
    return "Based on given selection method produce vector of parents for crossover operator";
}

template <class chromosome>
auto CrossoverParentSelectionStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population)
{
    if (!population.empty())
    {
        std::vector<geneticComponents::Parents<chromosome>> parents;
        parents.reserve(population.size());
        for (std::size_t i = 0; i < population.size(); i++)
        {
            parents.emplace_back(m_crossoverSelection.chooseParents(population));
        }

        return parents;
    }
    else
    {
        throw geneticComponents::PopulationIsEmptyException();
    }
}
} // namespace svmStrategies
