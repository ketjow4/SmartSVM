#pragma once

#include "libGeneticComponents/ISelectionOperator.h"

namespace geneticStrategies
{
template <class chromosome>
class CombinePopulationsStrategy
{
public:
    std::string getDescription() const;
    auto launch(geneticComponents::Population<chromosome>& population,
                geneticComponents::Population<chromosome>& other);
};

template <class chromosome>
std::string CombinePopulationsStrategy<chromosome>::getDescription() const
{
    return "Combines 2 populations into 1";
}

template <class chromosome>
auto CombinePopulationsStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population,
                                                    geneticComponents::Population<chromosome>& other)
{
    std::vector<chromosome> combined;
    combined.reserve(population.size() + other.size());
    combined.insert(combined.end(), population.begin(), population.end());
    combined.insert(combined.end(), other.begin(), other.end());

    return geneticComponents::Population<chromosome>(combined);
}
} // namespace svmStrategies
