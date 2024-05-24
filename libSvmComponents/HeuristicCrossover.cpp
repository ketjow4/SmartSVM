
#include <random>
#include "libGeneticComponents/Population.h"
#include "HeuristicCrossover.h"

namespace svmComponents
{
HeuristicCrossover::HeuristicCrossover(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                       std::uniform_real_distribution<double> alphaRange)
    : m_rngEngine(std::move(rngEngine))
    , m_alphaRange(std::move(alphaRange))
{
}

double HeuristicCrossover::getChildParameter(double lowFitnessParameter, double highFitnessParameter, double alpha)
{
    auto newParameter = lowFitnessParameter + alpha * (highFitnessParameter - lowFitnessParameter);
    auto iterationCount = 0;

    while(newParameter <= 0 && iterationCount++ < m_iterationLimit)
    {
        alpha = m_rngEngine->getRandom(m_alphaRange);
        newParameter = lowFitnessParameter + alpha * (highFitnessParameter - lowFitnessParameter);
    }

    if (newParameter <= 0)
    {
        return highFitnessParameter;
    }
    return newParameter;
}

HeuristicCrossover::chromosomeType HeuristicCrossover::crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB)
{
    auto minMaxPair = std::minmax(parentA, parentB);
    const auto& lowFitnessParent = minMaxPair.first;
    const auto& highFitnessParent = minMaxPair.second;

    auto alpha = m_rngEngine->getRandom(m_alphaRange);
    auto child = chromosomeType(parentA.getKernelType(), std::vector<double>(), parentA.isRegression());
    
    std::vector<double> newParameters(parentA.getKernelParameters().size());

    std::transform(lowFitnessParent.getKernelParameters().begin(),
                   lowFitnessParent.getKernelParameters().end(),
                   highFitnessParent.getKernelParameters().begin(),
                   newParameters.begin(),
                   [&](auto lowFitnessParameter, auto highFitnessParameter)
                   {
                       return getChildParameter(lowFitnessParameter, highFitnessParameter, alpha);
                   });

    child.updateKernelParameters(newParameters);
    return child;
}
} // namespace svmComponents