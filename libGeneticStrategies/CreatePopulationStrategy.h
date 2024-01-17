#pragma once

#include "libGeneticComponents/IPopulationGeneration.h"
#include "libGeneticComponents/GeneticExceptions.h"

namespace geneticStrategies
{
template <class chromosome>
class CreatePopulationStrategy
{
public:
    explicit CreatePopulationStrategy(geneticComponents::IPopulationGeneration<chromosome>& creation);

    std::string getDescription() const;
    auto launch(unsigned int populationSize);

private:
    geneticComponents::IPopulationGeneration<chromosome>& m_creation;
};

template <class chromosome>
CreatePopulationStrategy<chromosome>::CreatePopulationStrategy(geneticComponents::IPopulationGeneration<chromosome>& creation)
    : m_creation(creation)
{
}

template <class chromosome>
std::string CreatePopulationStrategy<chromosome>::getDescription() const
{
    return "Create population for genetic algorithm with given interface";
}

template <class chromosome>
auto CreatePopulationStrategy<chromosome>::launch(unsigned int populationSize)
{
    return m_creation.createPopulation(populationSize);
}
} // namespace svmStrategies
