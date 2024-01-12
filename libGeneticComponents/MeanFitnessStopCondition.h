
#pragma once

#include "LibGeneticComponents/ConsecutiveIterationFitnessBase.h"

namespace geneticComponents
{
template <typename chromosomeType>
class MeanFitnessStopCondition final : public ConsecutiveIterationFitnessBase<chromosomeType>
{
public:
    explicit MeanFitnessStopCondition(double epsilon);
    ~MeanFitnessStopCondition() override = default;

private:
    double getFitness(const Population<chromosomeType>& population) override;
};

template <typename chromosomeType>
MeanFitnessStopCondition<chromosomeType>::MeanFitnessStopCondition(double epsilon)
    : ConsecutiveIterationFitnessBase<chromosomeType>(epsilon)
{
}

template <typename chromosomeType>
double MeanFitnessStopCondition<chromosomeType>::getFitness(const Population<chromosomeType>& population)
{
    return population.getMeanFitness();
}
} // namespace geneticComponents
