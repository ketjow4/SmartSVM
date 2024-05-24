
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libPlatform/Percent.h"
#include "LibGeneticComponents/BitFlipMutation.h"
#include "LibGeneticComponents/GeneticUtils.h"

namespace geneticComponents
{
template <class binaryChromosome>
class FeaturesSelectionBitFlipMutation : public BitFlipMutation<binaryChromosome>
{
public:
    explicit FeaturesSelectionBitFlipMutation(platform::Percent flipProbability,
                                     std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine);

    void mutateChromosome(binaryChromosome& chromosome) override;

private:
    //logger::LogFrontend m_logger;
};

template <class binaryChromosome>
FeaturesSelectionBitFlipMutation<binaryChromosome>::FeaturesSelectionBitFlipMutation(platform::Percent flipProbability,
                                                                   std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine)
    : BitFlipMutation<binaryChromosome>(flipProbability, std::move(rngEngine))
{
}

template <class binaryChromosome>
void FeaturesSelectionBitFlipMutation<binaryChromosome>::mutateChromosome(binaryChromosome& chromosome)
{
    const auto genes = chromosome.getGenes(); 
    BitFlipMutation<binaryChromosome>::mutateChromosome(chromosome);

    if (geneticUtils::allZero(chromosome.getGenes()))
    {
        //m_logger.LOG(logger::LogLevel::Warning, "All bits of chromosome was 0 after mutation");
        chromosome.updateGenes(genes);
    }
}
} // namespace geneticComponents
