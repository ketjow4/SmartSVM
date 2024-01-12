
#pragma once

#include "LibGeneticComponents/ConsecutiveIterationFitnessBase.h"

namespace geneticComponents
{
template <typename chromosomeType>
class BestFitnessStopCondition final : public ConsecutiveIterationFitnessBase<chromosomeType>
{
public:
    explicit BestFitnessStopCondition(double epsilon);
    ~BestFitnessStopCondition() override = default;

private:
    double getFitness(const Population<chromosomeType>& population) override;
};

template <typename chromosomeType>
BestFitnessStopCondition<chromosomeType>::BestFitnessStopCondition(double epsilon)
    : ConsecutiveIterationFitnessBase<chromosomeType>(epsilon)
{
}

template <typename chromosomeType>
double BestFitnessStopCondition<chromosomeType>::getFitness(const Population<chromosomeType>& population)
{
    return population.getBestOne().getFitness();
}
} // namespace geneticComponents
