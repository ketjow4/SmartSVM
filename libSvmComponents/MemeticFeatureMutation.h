#pragma once
#include <memory>
#include <unordered_set>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libDataset/Dataset.h"
#include "libPlatform/Percent.h"
#include "SvmFeatureSetMemeticChromosome.h"
#include "SmallerPoolExperiment.h"


namespace svmComponents
{
// @wdudzik implementation based on: 2012 Kawulok and Nalepa - Support vector machines training data selection using a genetic algorithm
class MemeticFeatureMutation : public geneticComponents::IMutationOperator<SvmFeatureSetMemeticChromosome>
{
public:
    explicit MemeticFeatureMutation(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                    platform::Percent exchangePercent,
                                    platform::Percent mutationProbability,
                                    const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                    const std::vector<unsigned int>& labelsCount);

    void mutatePopulation(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population) override;
    void mutateChromosome(SvmFeatureSetMemeticChromosome& chromosome) override;

private:
    std::unordered_set<uint64_t> setDifference(const std::vector<Feature>& set, const std::unordered_set<uint64_t>& deleted);

    void calculateNumberOfPossibleExchanges(SvmFeatureSetMemeticChromosome& chromosome,
                                            uint64_t& possibleNumberOfExchangesPerClass) const;

    std::vector<Feature> findReplacement(const std::unordered_set<uint64_t>& deleted,
                                         std::unordered_set<uint64_t>& mutated,
                                         const std::vector<std::size_t>& positionsToReplace,
                                         SvmFeatureSetMemeticChromosome& chromosome) const;
    void getPositionsOfMutation(SvmFeatureSetMemeticChromosome& chromosome,
                                std::unordered_set<uint64_t>& deleted,
                                std::vector<std::size_t>& positionsToReplace) const;

    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    platform::Percent m_mutationProbability;
    platform::Percent m_exchangePercent;
    unsigned int m_numberOfExchanges;
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    unsigned int m_featureNumber;
};

inline MemeticFeatureMutation::MemeticFeatureMutation(std::unique_ptr<random::IRandomNumberGenerator> rngEngine, platform::Percent exchangePercent,
                                                      platform::Percent mutationProbability, const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                      const std::vector<unsigned>& /*labelsCount*/)
    : m_rngEngine(std::move(rngEngine))
    , m_mutationProbability(mutationProbability)
    , m_exchangePercent(exchangePercent)
    , m_numberOfExchanges(0)
    , m_trainingSet(trainingSet)
    , m_featureNumber(static_cast<unsigned int>(m_trainingSet.getSample(0).size()))
{
}

inline void MemeticFeatureMutation::mutatePopulation(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population)
{
    if (population.empty())
    {
        throw geneticComponents::PopulationIsEmptyException();
    }

    std::bernoulli_distribution mutation(m_mutationProbability.m_percentValue); //range 0% to 100%
    for (auto& chromosome : population)
    {
        if (m_rngEngine->getRandom(mutation))
        {
            mutateChromosome(chromosome);
        }
    }
}

inline void MemeticFeatureMutation::mutateChromosome(SvmFeatureSetMemeticChromosome& chromosome)
{
    std::unordered_set<uint64_t> deleted;
    std::vector<std::size_t> positionsToReplace;

    //@wdudzik in case of small number of features exchange at least one
    m_numberOfExchanges = std::max(static_cast<unsigned int>(std::floor(chromosome.size() * m_exchangePercent.m_percentValue)), 1u);
    positionsToReplace.reserve(m_numberOfExchanges);
    deleted.reserve(m_numberOfExchanges);

    // @wdudzik get values to be exchanged (mutated), and save into deleted
    getPositionsOfMutation(chromosome, deleted, positionsToReplace);

    //@wdudzik set difference between chromosome dataset and deleted
    auto mutated = setDifference(chromosome.getDataset(), deleted);

    //@wdudzik find and insert new ones. Restrictions are: no duplicates, cannot insert what was deleted, number of class examples have to match
    auto mutatedDataset = findReplacement(deleted, mutated, positionsToReplace, chromosome);
    chromosome.updateDataset(mutatedDataset);
}

inline std::unordered_set<uint64_t> MemeticFeatureMutation::setDifference(const std::vector<Feature>& set, const std::unordered_set<uint64_t>& deleted)
{
    std::vector<Feature> temporary;
    std::copy_if(set.begin(),
                 set.end(),
                 std::back_inserter(temporary),
                 [&deleted](const auto& featureVector)
                 {
                     return deleted.find(featureVector.id) == deleted.end();
                 });

    std::unordered_set<uint64_t> difference;
    difference.reserve(set.size());
    for (const auto& dataVector : temporary)
    {
        difference.insert(dataVector.id);
    }
    return difference;
}

inline void MemeticFeatureMutation::calculateNumberOfPossibleExchanges(SvmFeatureSetMemeticChromosome& chromosome,
   uint64_t& possibleNumberOfExchangesPerClass) const
{   
    if(SmallerPool::instance().size() <= chromosome.size())
    {
        possibleNumberOfExchangesPerClass = 0;
        return;
    }
    possibleNumberOfExchangesPerClass = SmallerPool::instance().size() - chromosome.size();
}

inline std::vector<Feature> MemeticFeatureMutation::findReplacement(const std::unordered_set<uint64_t>& deleted, std::unordered_set<uint64_t>& mutated,
                                                                    const std::vector<std::size_t>& positionsToReplace,
                                                                    SvmFeatureSetMemeticChromosome& chromosome) const
{
    std::uniform_int_distribution<int> featurePosition(0, static_cast<int>(m_trainingSet.getSample(0).size() - 1));

   

    auto dataset = chromosome.getDataset();

    uint64_t possibleNumberOfExchangesPerClass = 0;
    calculateNumberOfPossibleExchanges(chromosome, possibleNumberOfExchangesPerClass);

    for (auto i = 0u; i < m_numberOfExchanges; i++)
    {
        while (true)
        {
            auto newId = m_rngEngine->getRandom(featurePosition);
            if (SmallerPool::instance().isOk(newId) && deleted.find(newId) == deleted.end() &&
                mutated.emplace(newId).second) // @wdudzik is newId unique in chromosome dataset
            {
                dataset[positionsToReplace[i]].id = newId;
                possibleNumberOfExchangesPerClass--;
                break;
            }
            if (possibleNumberOfExchangesPerClass == 0 || possibleNumberOfExchangesPerClass > m_trainingSet.getSample(0).size())
            {
                break;
            }
        }
    }
    return dataset;
}

inline void MemeticFeatureMutation::getPositionsOfMutation(SvmFeatureSetMemeticChromosome& chromosome,
                                                           std::unordered_set<uint64_t>& deleted,
                                                           std::vector<std::size_t>& positionsToReplace) const
{
    auto& features = chromosome.getDataset();
    std::uniform_int_distribution<int> replacePosition(0, static_cast<int>(features.size() - 1));
    for (auto i = 0u; i < m_numberOfExchanges;)
    {
        auto position = m_rngEngine->getRandom(replacePosition);
        if (deleted.insert(features[position].id).second) // @wdudzik if position is unique
        {
            positionsToReplace.emplace_back(position);
            ++i;
        }
    }
}
} // namespace svmComponents
