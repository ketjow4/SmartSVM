
#include "libGeneticComponents/Population.h"
#include "GaSvmMutation.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "SvmTrainingSetChromosome.h"

namespace svmComponents
{
GaSvmMutation::GaSvmMutation(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                             platform::Percent exchangePercent,
                             platform::Percent mutationProbability,
                             const dataset::Dataset<std::vector<float>, float>& trainingSet,
                             const std::vector<unsigned int>& labelsCount)
    : m_rngEngine(std::move(rngEngine))
    , m_mutationProbability(mutationProbability)
    , m_exchangePercent(exchangePercent)
    , m_numberOfExchanges(0)
    , m_numberOfClasses(static_cast<unsigned int>(labelsCount.size()))
    , m_labelsCount(labelsCount)
    , m_trainingSet(trainingSet)
{
}

void GaSvmMutation::mutatePopulation(geneticComponents::Population<SvmTrainingSetChromosome>& population)
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

std::unordered_set<uint64_t> GaSvmMutation::setDifference(const std::vector<DatasetVector>& set,
                                                          const std::unordered_set<uint64_t>& deleted)
{
    std::vector<DatasetVector> temporary;
    std::copy_if(set.begin(),
                 set.end(),
                 std::back_inserter(temporary),
                 [&deleted](const auto& dataVector)
         {
             return deleted.find(dataVector.id) == deleted.end();
         });

    std::unordered_set<uint64_t> difference;
    difference.reserve(set.size());
    for (const auto& dataVector : temporary)
    {
        difference.insert(dataVector.id);
    }
    return difference;
}

void GaSvmMutation::calculateNumberOfPossibleExchanges(SvmTrainingSetChromosome& chromosome,
                                                       std::vector<uint64_t>& possibleNumberOfExchangesPerClass) const
{
    for(auto i = 0u; i < m_numberOfClasses; ++i)
    {
        int test = static_cast<int>(static_cast<int>(m_labelsCount[i]) - (chromosome.size()) / m_numberOfClasses);
    	if (test <= 0)
    	{
            possibleNumberOfExchangesPerClass[i] = 0;
    	}
        else
        {
	        possibleNumberOfExchangesPerClass[i] = m_labelsCount[i] - (chromosome.size() / m_numberOfClasses);
        }
    }
}

std::vector<DatasetVector> GaSvmMutation::findReplacement(const std::unordered_set<uint64_t>& deleted,
                                                          std::unordered_set<uint64_t>& mutated,
                                                          const std::vector<std::size_t>& positionsToReplace,
                                                          SvmTrainingSetChromosome& chromosome) const
{
    std::uniform_int_distribution<int> datasetPosition(0, static_cast<int>(m_trainingSet.size() - 1));
    const auto targets = m_trainingSet.getLabels();
    auto dataset = chromosome.getDataset();

    std::vector<uint64_t> possibleNumberOfExchangesPerClass(m_numberOfClasses);
    calculateNumberOfPossibleExchanges(chromosome, possibleNumberOfExchangesPerClass);

    for (auto i = 0u; i < m_numberOfExchanges; i++)
    {
        auto j = 0;
        while (true && j < 10000)
        {
            auto newId = m_rngEngine->getRandom(datasetPosition);
            if (static_cast<int>(dataset[positionsToReplace[i]].classValue) == targets[newId] && // @wdudzik class value match
                deleted.find(newId) == deleted.end() &&
                mutated.emplace(newId).second) // @wdudzik is newId unique in chromosome dataset
            {
                dataset[positionsToReplace[i]].id = newId;
                possibleNumberOfExchangesPerClass[static_cast<int>(dataset[positionsToReplace[i]].classValue)]--;
                break;
            }
            if (possibleNumberOfExchangesPerClass[static_cast<int>(dataset[positionsToReplace[i]].classValue)] == 0)
            {
                break;
            }
            j++;
        }
    }
    return dataset;
}

void GaSvmMutation::getPositionsOfMutation(SvmTrainingSetChromosome& chromosome,
                                           std::unordered_set<uint64_t>& deleted,
                                           std::vector<std::size_t>& positionsToReplace) const
{
    auto& dataset = chromosome.getDataset();
    std::uniform_int_distribution<int> replacePosition(0, static_cast<int>(dataset.size() - 1));
    for (auto i = 0u; i < m_numberOfExchanges;)
    {
        auto position = m_rngEngine->getRandom(replacePosition);
        if (deleted.insert(dataset[position].id).second) // @wdudzik if position is unique
        {
            positionsToReplace.emplace_back(position);
            ++i;
        }
    }
}

void GaSvmMutation::mutateChromosome(SvmTrainingSetChromosome& chromosome)
{
    std::unordered_set<uint64_t> deleted;
    std::vector<std::size_t> positionsToReplace;

    m_numberOfExchanges = static_cast<unsigned int>(std::floor(chromosome.size() * m_exchangePercent.m_percentValue));
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
} // namespace svmComponents
