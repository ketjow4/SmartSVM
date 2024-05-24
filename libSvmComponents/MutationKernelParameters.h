
#pragma once
#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libPlatform/Percent.h"

namespace svmComponents
{
class MutationKernelParameters : public geneticComponents::IMutationOperator<SvmKernelChromosome>
{
public:
    explicit MutationKernelParameters(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                      platform::Percent maxPercentChange,
                                      platform::Percent mutationProbability);

    void mutatePopulation(geneticComponents::Population<SvmKernelChromosome>& population) override;
    void mutateChromosome(SvmKernelChromosome& chromosome) override;

private:
    int getRandomSign() const;

    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    platform::Percent m_mutationProbability;
    platform::Percent m_maxPercentChange;
};
} // namespace svmComponents
