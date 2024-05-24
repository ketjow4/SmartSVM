#pragma once

#include <memory>
#include "libGeneticComponents/ICrossoverSelection.h"
#include "libRandom/MersenneTwister64Rng.h"
#include "libPlatform/Percent.h"

namespace geneticComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
template <class chromosome>
class LocalGlobalAdaptationSelection : public ICrossoverSelection<chromosome>
{
public:
	explicit LocalGlobalAdaptationSelection(bool isLocalMode,
	                                        platform::Percent highLowCoefficient,
	                                        std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine);

	Parents<chromosome> chooseParents(Population<chromosome>& population) override;

	void setMode(bool isLocal);

	Indexes chooseIndexes(Population<chromosome>& population) override;

private:
	Indexes localModeSelection(Population<chromosome>& population,
	                           std::uniform_int_distribution<int> lowPart,
	                           std::uniform_int_distribution<int> highPart);
	Indexes globalModeSelection(Population<chromosome>& population,
	                            std::uniform_int_distribution<int> lowPart,
	                            std::uniform_int_distribution<int> highPart);

	bool m_isLocalMode;
	platform::Percent m_highLowCoefficient;
	unsigned int m_count;
	std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
};

template <class chromosome>
LocalGlobalAdaptationSelection<chromosome>::LocalGlobalAdaptationSelection(bool isLocalMode,
                                                                           platform::Percent highLowCoefficient,
                                                                           std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine)
	: m_isLocalMode(isLocalMode)
	, m_highLowCoefficient(highLowCoefficient)
	, m_rngEngine(std::move(rngEngine))
	, m_count(0)
{
}

#pragma warning( push )
#pragma warning( disable : 4505)
// @wdudzik Warning C4505 unreferenced local function - there is situation in which I use only setMode method directly in code

template <class chromosome>
Indexes LocalGlobalAdaptationSelection<chromosome>::localModeSelection(Population<chromosome>& population,
                                                                       std::uniform_int_distribution<int> lowPart,
                                                                       std::uniform_int_distribution<int> highPart)
{
	m_count = (m_count + 1) % population.size();

	if (m_count < std::floor(m_highLowCoefficient.m_percentValue * population.size()))
	{
		//better part of population
		auto parentA = m_rngEngine->getRandom(highPart);
		auto parentB = m_rngEngine->getRandom(highPart);

		//return Parents<chromosome>(population[parentA], population[parentB]);
		return Indexes(parentA, parentB);
	}
	else
	{
		//worse part of population
		auto parentA = m_rngEngine->getRandom(lowPart);
		auto parentB = m_rngEngine->getRandom(lowPart);

		// return Parents<chromosome>(population[parentA], population[parentB]);
		return Indexes(parentA, parentB);
	}
}

template <class chromosome>
Indexes LocalGlobalAdaptationSelection<chromosome>::globalModeSelection(Population<chromosome>& /*population*/,
                                                                        std::uniform_int_distribution<int> lowPart,
                                                                        std::uniform_int_distribution<int> highPart)
{
	auto high = m_rngEngine->getRandom(highPart);
	auto low = m_rngEngine->getRandom(lowPart);

	//return Parents<chromosome>(population[high], population[low]);
	return Indexes(high, low);
}

template <class chromosome>
Parents<chromosome> LocalGlobalAdaptationSelection<chromosome>::chooseParents(Population<chromosome>& population)
{
	/*if (!std::is_sorted(population.begin(), population.end()))
	{
		std::sort(population.begin(), population.end());
	}

	auto highLowCut = static_cast<int>(std::floor(m_highLowCoefficient.m_percentValue * population.size()));
	constexpr auto beginIndex = 0;
	auto endIndex = population.size() - 1;
	std::uniform_int_distribution<int> lowPart(beginIndex, highLowCut);
	std::uniform_int_distribution<int> highPart(highLowCut, static_cast<int>(endIndex));

	if (m_isLocalMode)
	{
		return localModeSelection(population, lowPart, highPart);
	}
	else
	{
		return globalModeSelection(population, lowPart, highPart);
	}*/

	auto indexes = chooseIndexes(population);

	return Parents<chromosome>((population[indexes.first]), (population[indexes.second]));
}

#pragma warning( pop )

template <class chromosome>
void LocalGlobalAdaptationSelection<chromosome>::setMode(bool isLocal)
{
	m_isLocalMode = isLocal;
	m_count = 0;
}

template <class chromosome>
Indexes LocalGlobalAdaptationSelection<chromosome>::chooseIndexes(Population<chromosome>& population)
{
	if (!std::is_sorted(population.begin(), population.end()))
	{
		std::sort(population.begin(), population.end());
	}

	auto highLowCut = static_cast<int>(std::floor(m_highLowCoefficient.m_percentValue * population.size()));
	constexpr auto beginIndex = 0;
	auto endIndex = population.size() - 1;
	std::uniform_int_distribution<int> lowPart(beginIndex, highLowCut);
	std::uniform_int_distribution<int> highPart(highLowCut, static_cast<int>(endIndex));

	if (m_isLocalMode)
	{
		return localModeSelection(population, lowPart, highPart);
	}
	else
	{
		return globalModeSelection(population, lowPart, highPart);
	}
}
} // namespace geneticComponents
