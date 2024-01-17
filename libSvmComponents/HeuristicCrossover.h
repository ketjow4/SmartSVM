
#pragma once

#include <memory>
#include <random>
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/BaseCrossoverOperator.h"
#include "libSvmComponents/SvmKernelChromosome.h"

namespace svmComponents
{
class HeuristicCrossover : public geneticComponents::BaseCrossoverOperator<SvmKernelChromosome>
{
public:
    using chromosomeType = SvmKernelChromosome;

    explicit HeuristicCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                std::uniform_real_distribution<double> alphaRange);

    chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB) override;

private:
    // @wdudzik aplha default value comes from: Kaya et al., A Novel Crossover Operator for Genetic Algorithms: Ring Crossover, arXiv:1105.0355, 2011.
    double getChildParameter(double lowFitnessParameter, double highFitnessParameter, double alpha = 1.2);

    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    std::uniform_real_distribution<double> m_alphaRange;
    static constexpr auto m_iterationLimit = 100U;
};
} // namespace svmComponents
