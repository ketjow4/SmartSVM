
#pragma once
#include <memory>
#include <unordered_set>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libDataset/Dataset.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libPlatform/Percent.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2012 Kawulok and Nalepa - Support vector machines training data selection using a genetic algorithm
class GaSvmMutation : public geneticComponents::IMutationOperator<SvmTrainingSetChromosome>
{
public:
    explicit GaSvmMutation(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                           platform::Percent exchangePercent,
                           platform::Percent mutationProbability,
                           const dataset::Dataset<std::vector<float>, float>& trainingSet,
                           const std::vector<unsigned int>& labelsCount);

    void mutatePopulation(geneticComponents::Population<SvmTrainingSetChromosome>& population) override;
    void mutateChromosome(SvmTrainingSetChromosome& chromosome) override;

private:
    std::unordered_set<uint64_t> setDifference(const std::vector<DatasetVector>& set, const std::unordered_set<uint64_t>& deleted);
    void calculateNumberOfPossibleExchanges(SvmTrainingSetChromosome& chromosome,
                                            std::vector<uint64_t>& possibleNumberOfExchangesPerClass) const;
    std::vector<DatasetVector> findReplacement(const std::unordered_set<uint64_t>& deleted,
                                               std::unordered_set<uint64_t>& mutated,
                                               const std::vector<std::size_t>& positionsToReplace,
                                               SvmTrainingSetChromosome& chromosome) const;
    void getPositionsOfMutation(SvmTrainingSetChromosome& chromosome,
                                std::unordered_set<uint64_t>& deleted,
                                std::vector<std::size_t>& positionsToReplace) const;

    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    platform::Percent m_mutationProbability;
    platform::Percent m_exchangePercent;
    unsigned int m_numberOfExchanges;
    const unsigned int m_numberOfClasses;
    const std::vector<unsigned int> m_labelsCount;
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
};
} // namespace svmComponents
