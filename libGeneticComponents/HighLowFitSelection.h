
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/ICrossoverSelection.h"
#include "LibGeneticComponents/GeneticExceptions.h"
#include "libPlatform/Percent.h"

namespace geneticComponents
{
template <typename T>
class HighLowFitSelection : public ICrossoverSelection<T>
{
public:
    explicit HighLowFitSelection(platform::Percent highLowCoefficient,
                                 std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine);

    Parents<T> chooseParents(Population<T>& population) override;

    Indexes chooseIndexes(Population<T>& population) override;
	
private:
    platform::Percent m_highLowCoefficient;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
};

template <typename T>
HighLowFitSelection<T>::HighLowFitSelection(platform::Percent highLowCoefficient,
                                            std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine)
    : m_highLowCoefficient(highLowCoefficient)
    , m_rngEngine(std::move(rngEngine))
{
}

template <typename T>
Parents<T> HighLowFitSelection<T>::chooseParents(Population<T>& population)
{
	//THIS CODE IS MOVED TO ANOTHER FUNCTION
    //if (population.empty())
    //{
    //    throw PopulationIsEmptyException();
    //}
    //if (!std::is_sorted(population.begin(), population.end()))
    //{
    //    std::sort(population.begin(), population.end());
    //}

    //auto highLowCut = static_cast<int>(std::floor(m_highLowCoefficient.m_percentValue * population.size()));
    //auto endIndex = population.size() - 1;
    //constexpr auto beginIndex = 0;

    //std::uniform_int_distribution<int> lowPart(beginIndex, highLowCut);
    //std::uniform_int_distribution<int> highPart(highLowCut, static_cast<int>(endIndex));

    //auto high = m_rngEngine->getRandom(highPart);
    //auto low = m_rngEngine->getRandom(lowPart);

    //return Parents<T>((population[high]), (population[low]));

    auto indexes = chooseIndexes(population);

	return Parents<T>((population[indexes.first]), (population[indexes.second]));
	
}

template <typename T>
Indexes HighLowFitSelection<T>::chooseIndexes(Population<T>& population)
{
    if (population.empty())
    {
        throw PopulationIsEmptyException();
    }
    if (!std::is_sorted(population.begin(), population.end()))
    {
        std::sort(population.begin(), population.end());
    }

    auto highLowCut = static_cast<int>(std::floor(m_highLowCoefficient.m_percentValue * population.size()));
    auto endIndex = population.size() - 1;
    constexpr auto beginIndex = 0;

    std::uniform_int_distribution<int> lowPart(beginIndex, highLowCut);
    std::uniform_int_distribution<int> highPart(highLowCut, static_cast<int>(endIndex));

    auto high = m_rngEngine->getRandom(highPart);
    auto low = m_rngEngine->getRandom(lowPart);

    return Indexes(high, low);
}
} // namespace geneticComponents
