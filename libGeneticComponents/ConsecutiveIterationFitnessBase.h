
#pragma once

#include "LibGeneticComponents/GeneticExceptions.h"
#include "LibGeneticComponents/IStopCondition.h"

namespace geneticComponents
{
template <typename chromosomeType>
class ConsecutiveIterationFitnessBase : public IStopCondition<chromosomeType>
{
public:
    explicit ConsecutiveIterationFitnessBase(double epsilon);

    bool isFinished(const Population<chromosomeType>& population) override;

    void reset() override;

protected:
     virtual double getFitness(const Population<chromosomeType>& population) = 0;


private:
    double m_epsilon;
    double m_previousFitness;
    static constexpr auto uninitializedFitness = -1.0;
};

template <typename chromosomeType>
ConsecutiveIterationFitnessBase<chromosomeType>::ConsecutiveIterationFitnessBase(double epsilon)
    : m_epsilon(epsilon)
    , m_previousFitness(uninitializedFitness)
{
}

template <typename chromosomeType>
bool ConsecutiveIterationFitnessBase<chromosomeType>::isFinished(const Population<chromosomeType>& population)
{
    if (population.empty())
    {
        throw PopulationIsEmptyException();
    }
    auto fitness = getFitness(population);

    if (std::fabs(fitness - m_previousFitness) < m_epsilon)
    {
        return true;
    }
    m_previousFitness = fitness;
    return false;
}

template <typename chromosomeType>
void ConsecutiveIterationFitnessBase<chromosomeType>::reset()
{
    m_previousFitness = uninitializedFitness;
}
} // namespace geneticComponents
#pragma once
