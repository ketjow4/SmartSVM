#pragma once

#include "libGeneticComponents/IStopCondition.h"
#include "libGeneticComponents/GeneticExceptions.h"

namespace geneticStrategies
{
template <class chromosome>
class StopConditionStrategy
{
public:
    StopConditionStrategy(geneticComponents::IStopCondition<chromosome>& stopCondition);

    std::string getDescription() const;
    bool launch(geneticComponents::Population<chromosome>& population);

    void reset() const
    {
        m_stopCondition.reset();
    }

private:
    geneticComponents::IStopCondition<chromosome>& m_stopCondition;
};

template <class chromosome>
StopConditionStrategy<chromosome>::StopConditionStrategy(geneticComponents::IStopCondition<chromosome>& stopCondition)
    : m_stopCondition(stopCondition)
{
}

template <class chromosome>
std::string StopConditionStrategy<chromosome>::getDescription() const
{
    return "Check if stop conditions for genetic algorithm is satisfied";
}

template <class chromosome>
bool StopConditionStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population)
{
    return m_stopCondition.isFinished(population);
}
} // namespace svmStrategies
