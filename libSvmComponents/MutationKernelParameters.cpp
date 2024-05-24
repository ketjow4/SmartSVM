
#include <random>
#include "libGeneticComponents/GeneticExceptions.h"
#include "libGeneticComponents/Population.h"
#include "MutationKernelParameters.h"
#include "SvmUtils.h"

namespace svmComponents
{
MutationKernelParameters::MutationKernelParameters(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                                   platform::Percent maxPercentChange,
                                                   platform::Percent mutationProbability)
    : m_rngEngine(std::move(rngEngine))
    , m_mutationProbability(mutationProbability)
    , m_maxPercentChange(maxPercentChange)
{
}

void MutationKernelParameters::mutatePopulation(geneticComponents::Population<SvmKernelChromosome>& population)
{
    if (population.empty())
    {
        throw geneticComponents::PopulationIsEmptyException();
    }

    std::bernoulli_distribution mutation(m_mutationProbability.m_percentValue);
    std::sort(population.begin(), population.end());

    for (auto& chromosome : population)
    {
        if (m_rngEngine->getRandom(mutation))
        {
            mutateChromosome(chromosome);
        }
    }
}

void MutationKernelParameters::mutateChromosome(SvmKernelChromosome& chromosome)
{
    std::uniform_real_distribution<double> mutation(platform::Percent::m_minPercent, m_maxPercentChange.m_percentValue);

    auto newParameters(chromosome.getKernelParameters());
    auto oldParameters = chromosome.getKernelParameters();
    
    std::transform(oldParameters.begin(), 
        oldParameters.end(), 
        newParameters.begin(),
        newParameters.begin(), 
        [&,this](auto oldParameter, auto newParameter)
    {
        auto sign = getRandomSign();
        auto newValue = m_rngEngine->getRandom(mutation) * (oldParameter * sign) + newParameter;
        if(newValue <= 0)
        {
            return oldParameter;
        }
        return newValue;
    });
    chromosome.updateKernelParameters(newParameters);
}

int MutationKernelParameters::getRandomSign() const
{
    std::uniform_real_distribution<double> sign(platform::Percent::m_minPercent, platform::Percent::m_maxPercent);
    constexpr double halfRange = platform::Percent::m_maxPercent / 2;
    constexpr auto positive = 1;
    constexpr auto negative = -1;
    return m_rngEngine->getRandom(sign) > halfRange ? positive : negative;
}
} // namespace svmComponents
