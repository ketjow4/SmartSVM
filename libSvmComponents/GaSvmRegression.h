
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/TraningSetCrossover.h"
#include <unordered_set>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libDataset/Dataset.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libPlatform/Percent.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libGeneticComponents/Population.h"
#include "GaSvmMutation.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "libPlatform/StringUtils.h"

namespace svmComponents
{
    class ValidInTrain
    {
    private:
        ValidInTrain()
        {
            setupMask();
        };

        std::unordered_set<uint32_t> validInTrainIndex;

    public:
        static ValidInTrain& instance()
        {
            static ValidInTrain INSTANCE;
            return INSTANCE;
        }

        void setupMask()
        {
            // std::ifstream ifs(R"(D:\datasetsFolds2\CLASH\valid_in_train.txt)");
            // std::string content((std::istreambuf_iterator<char>(ifs)),
            //                     (std::istreambuf_iterator<char>()));

            // auto indexes = platform::stringUtils::splitString(content, ',');
            // for(const auto& i : indexes)
            // {
            //     validInTrainIndex.emplace(std::stoi(i));
            // }
        }

        size_t size()
        {
            return validInTrainIndex.size();
        }

        bool isOk(uint32_t id)
        {
            //return true;
            return validInTrainIndex.find(id) != validInTrainIndex.end();
        }
    };




// @wdudzik implementation based on: 2012 Kawulok and Nalepa - Support vector machines training data selection using a genetic algorithm
class GaSvmCrossoverRegression : public TrainingSetCrossover
{
public:
    using chromosomeType = SvmTrainingSetChromosome;

    explicit GaSvmCrossoverRegression(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                      unsigned int numberOfClasses);

    chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB) override;

private:
    std::vector<DatasetVector> crossoverInternals2(const chromosomeType& parentA,
                                                   const chromosomeType& parentB,
                                                   unsigned int datasetSize,
                                                   unsigned int classExamples);

    void tryToInsert2(const std::vector<DatasetVector>::const_iterator& parentAIt,
                      const std::vector<DatasetVector>::const_iterator& parentBIt,
                      std::unordered_set<uint64_t>& childSet,
                      std::vector<unsigned>& classes,
                      unsigned classExamples,
                      std::vector<DatasetVector>& child) const;
};

class GaSvmGenerationRegression : public geneticComponents::IPopulationGeneration<SvmTrainingSetChromosome>
{
public:
    explicit GaSvmGenerationRegression(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                       std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                       unsigned int numberOfClassExamples,
                                       const std::vector<unsigned int>& labelsCount);

    geneticComponents::Population<SvmTrainingSetChromosome> createPopulation(uint32_t populationSize) override;

private:
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClassExamples;
    unsigned int m_numberOfClasses;
};

class GaSvmMutationRegression : public geneticComponents::IMutationOperator<SvmTrainingSetChromosome>
{
public:
    explicit GaSvmMutationRegression(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
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

    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    platform::Percent m_mutationProbability;
    platform::Percent m_exchangePercent;
    unsigned int m_numberOfExchanges;
    const unsigned int m_numberOfClasses;
    const std::vector<unsigned int> m_labelsCount;
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
};

inline GaSvmGenerationRegression::GaSvmGenerationRegression(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                            std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                                            unsigned int numberOfClassExamples,
                                                            const std::vector<unsigned int>& labelsCount)
    : m_trainingSet(trainingSet)
    , m_rngEngine(std::move(rngEngine))
    , m_numberOfClassExamples(numberOfClassExamples)
    , m_numberOfClasses(static_cast<unsigned int>(labelsCount.size()))
{
    /* if (std::any_of(labelsCount.begin(), labelsCount.end(), [this](const auto& labelCount) { return m_numberOfClassExamples > labelCount;  }))
     {
         throw ValueOfClassExamplesIsTooHighForDataset(m_numberOfClassExamples);
     }*/
}

inline geneticComponents::Population<SvmTrainingSetChromosome> GaSvmGenerationRegression::createPopulation(uint32_t populationSize)
{
    if (populationSize == 0)
    {
        throw geneticComponents::PopulationIsEmptyException();
    }

    auto targets = m_trainingSet.getLabels();
    auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));
    std::vector<SvmTrainingSetChromosome> population(populationSize);

    std::generate(population.begin(), population.end(), [&]
    {
        std::unordered_set<std::uint64_t> trainingSet;
        std::vector<DatasetVector> chromosomeDataset;
        chromosomeDataset.reserve(m_numberOfClassExamples * 1);
        auto howMany = 0u;
        while (howMany < m_numberOfClassExamples)
        {
            auto randomValue = m_rngEngine->getRandom(trainingSetID);
            
            if (/*(targets[randomValue] < 0.4 || targets[randomValue] > 0.6) &&*/
                ValidInTrain::instance().isOk(static_cast<uint32_t>(randomValue)) &&
                trainingSet.emplace(static_cast<int>(randomValue)).second) // is unique
            {
                chromosomeDataset.emplace_back(DatasetVector(randomValue, targets[randomValue]));
                howMany++;
            }
        }
        return SvmTrainingSetChromosome(std::move(chromosomeDataset));
    });
    return geneticComponents::Population<SvmTrainingSetChromosome>(std::move(population));
}

inline GaSvmMutationRegression::GaSvmMutationRegression(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
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

inline void GaSvmMutationRegression::mutatePopulation(geneticComponents::Population<SvmTrainingSetChromosome>& population)
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

inline std::unordered_set<uint64_t> GaSvmMutationRegression::setDifference(const std::vector<DatasetVector>& set,
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

inline void GaSvmMutationRegression::calculateNumberOfPossibleExchanges(SvmTrainingSetChromosome& chromosome,
                                                                        std::vector<uint64_t>& possibleNumberOfExchangesPerClass) const
{
    for (auto i = 0u; i < m_numberOfClasses; ++i)
    {
        possibleNumberOfExchangesPerClass[i] = m_labelsCount[i] - (chromosome.size() / m_numberOfClasses);
    }
}

inline std::vector<DatasetVector> GaSvmMutationRegression::findReplacement(const std::unordered_set<uint64_t>& deleted,
                                                                           std::unordered_set<uint64_t>& mutated,
                                                                           const std::vector<std::size_t>& positionsToReplace,
                                                                           SvmTrainingSetChromosome& chromosome) const
{
    std::uniform_int_distribution<int> datasetPosition(0, static_cast<int>(m_trainingSet.size() - 1));
    const auto targets = m_trainingSet.getLabels();
    auto dataset = chromosome.getDataset();

    for (auto i = 0u; i < m_numberOfExchanges; i++)
    {
        while (true)
        {
            auto newId = m_rngEngine->getRandom(datasetPosition);
            if (ValidInTrain::instance().isOk(static_cast<uint32_t>(newId)) &&
                deleted.find(newId) == deleted.end() &&
                mutated.emplace(newId).second) // @wdudzik is newId unique in chromosome dataset
            {
                dataset[positionsToReplace[i]].id = newId;
                break;
            }
        }
    }
    return dataset;
}

inline void GaSvmMutationRegression::getPositionsOfMutation(SvmTrainingSetChromosome& chromosome,
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

inline void GaSvmMutationRegression::mutateChromosome(SvmTrainingSetChromosome& chromosome)
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

inline GaSvmCrossoverRegression::GaSvmCrossoverRegression(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                                          unsigned int numberOfClasses)
    : TrainingSetCrossover(std::move(rngEngine), numberOfClasses)
{
    /*if (numberOfClasses < 2)
    {
        throw TooSmallNumberOfClasses(numberOfClasses);
    }*/
}

inline GaSvmCrossoverRegression::chromosomeType GaSvmCrossoverRegression::crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB)
{
    if (parentA.getDataset().size() != parentB.getDataset().size())
    {
        throw CrossoverParentsSizeInequality(parentA.getDataset().size(), parentB.getDataset().size());
    }

    auto datasetSize = static_cast<unsigned int>(parentA.getDataset().size());
    auto classExamples = static_cast<unsigned int>(datasetSize / 1);

    auto child = crossoverInternals2(parentA, parentB, datasetSize, classExamples);
    return SvmTrainingSetChromosome(std::move(child));
}

inline void GaSvmCrossoverRegression::tryToInsert2(const std::vector<DatasetVector>::const_iterator& parentAIt,
                                                   const std::vector<DatasetVector>::const_iterator& parentBIt,
                                                   std::unordered_set<uint64_t>& childSet,
                                                   std::vector<unsigned>& /*classes*/,
                                                   unsigned /*classExamples*/,
                                                   std::vector<DatasetVector>& child) const
{
    if (childSet.insert(parentAIt->id).second)
    {
        child.emplace_back(*parentAIt);
    }
    else if (childSet.insert(parentBIt->id).second)
    {
        child.emplace_back(*parentBIt);
    }
}

inline std::vector<DatasetVector> GaSvmCrossoverRegression::crossoverInternals2(const chromosomeType& parentA,
                                                                                const chromosomeType& parentB,
                                                                                unsigned int datasetSize,
                                                                                unsigned int classExamples)
{
    std::vector<DatasetVector> child;
    std::vector<unsigned int> classes;
    std::unordered_set<uint64_t> childSet;
    child.reserve(datasetSize);
    childSet.reserve(datasetSize);
    classes.resize(m_numberOfClasses);

    auto parentAIt = parentA.getDataset().begin();
    auto parentBIt = parentB.getDataset().begin();

    while (true)
    {
        std::uniform_real_distribution<double> randomParent(platform::Percent::m_minPercent, platform::Percent::m_maxPercent);
        constexpr auto halfRange = platform::Percent::m_maxPercent / 2;
        auto parentChoose = m_rngEngine->getRandom(randomParent);
        if (parentChoose > halfRange)
        {
            tryToInsert(parentAIt, parentBIt, childSet, classes, classExamples, child);
        }
        else
        {
            tryToInsert(parentBIt, parentAIt, childSet, classes, classExamples, child);
        }

        if (childSet.size() == classExamples)
        {
            break;
        }

        if (++parentBIt == parentB.getDataset().end())
        {
            parentBIt = parentB.getDataset().begin();
        }
        if (++parentAIt == parentA.getDataset().end())
        {
            parentAIt = parentA.getDataset().begin();
        }
    }
    return child;
}
} // namespace svmComponents

