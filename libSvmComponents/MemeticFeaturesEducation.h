#pragma once
#include <memory>
#include "LibGeneticComponents/ICrossoverSelection.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "libSvmComponents/SupportVectorPool.h"
#include "libSvmComponents/SvmUtils.h"
#include "libPlatform/Percent.h"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticFeaturesEducation
{
public:
    virtual ~MemeticFeaturesEducation() = default;

    MemeticFeaturesEducation(platform::Percent educationProbability,
                             unsigned int numberOfClasses,
                             std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator,
                             std::unique_ptr<IFeatureSelection> supportVectorSelection);

    void educatePopulation(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                           const std::vector<Feature>& supportVectorPool,
                           const std::vector<geneticComponents::Parents<SvmFeatureSetMemeticChromosome>>& parents,
                           const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

    void educate(SvmFeatureSetMemeticChromosome& individual,
                 const geneticComponents::Parents<SvmFeatureSetMemeticChromosome>& parents,
                 const std::vector<Feature>& supportVectorPool,
                 const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

private:
    static bool replacementCondition(const Feature& supportVectorPoolElement,
                                     std::unordered_set<std::uint64_t>& traningIDs,
                                     const Feature& sample);

    static std::vector<Feature> setDifference(const std::unordered_set<uint64_t>& svPool,
                                              const std::vector<Feature>& traningDataset);

    platform::Percent m_educationProbability;
    const unsigned int m_numberOfClasses;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    std::unique_ptr<IFeatureSelection> m_featureSelection;
};

inline MemeticFeaturesEducation::MemeticFeaturesEducation(platform::Percent educationProbability, unsigned numberOfClasses,
                                                          std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator,
                                                          std::unique_ptr<IFeatureSelection> supportVectorSelection)
    : m_educationProbability(educationProbability)
    , m_numberOfClasses(numberOfClasses)
    , m_rngEngine(std::move(randomNumberGenerator))
    , m_featureSelection(std::move(supportVectorSelection))
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
    if (m_featureSelection == nullptr)
    {
        throw MemberNullPointer("m_supportVectorSelection");
    }
}

inline void MemeticFeaturesEducation::educatePopulation(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                                                        const std::vector<Feature>& featurePool,
                                                        const std::vector<geneticComponents::Parents<SvmFeatureSetMemeticChromosome>>& parents,
                                                        const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
    auto parentIterator = parents.begin();
    for (auto& individual : population)
    {
        educate(individual, *parentIterator, featurePool, trainingSet);
        ++parentIterator;
    }
}

inline std::unordered_set<uint64_t> sumFeaturesSet(const geneticComponents::Parents<SvmFeatureSetMemeticChromosome>& parents)
{
    auto featureSetOfParents = parents.first.convertToSet();

    for (auto feature : parents.second.convertToSet())
    {
        featureSetOfParents.emplace(feature);
    }

    return featureSetOfParents;
}

inline void MemeticFeaturesEducation::educate(SvmFeatureSetMemeticChromosome& individual,
                                              const geneticComponents::Parents<SvmFeatureSetMemeticChromosome>& parents,
                                              const std::vector<Feature>& featurePool,
                                              const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/) const
{
    auto features = individual.getDataset();
    auto parentsFeatures = sumFeaturesSet(parents);
    auto weakFeatures = setDifference(parentsFeatures, features);
    auto featuresIDs = individual.convertToSet();

    //@wdudzik featurePool - featuresIDs; possible replecement for weakFeatures
    auto result = setDifference(featuresIDs, featurePool);

    if (result.empty())
    {
        return;
    }

    auto numberOfPossibleExchanges = result.size();
    std::bernoulli_distribution education(m_educationProbability.m_percentValue);
    const auto endIndex = static_cast<int>(featurePool.size() - 1);
    std::uniform_int_distribution<int> featurePoolID(0, endIndex);

    for (const auto& sample : weakFeatures)
    {
        if (m_rngEngine->getRandom(education))
        {
            auto positionInDataset = std::find(features.begin(), features.end(), sample);

            //@wdudzik try to replace with one form supportVectorPool
            while (true)
            {
                auto newId = m_rngEngine->getRandom(featurePoolID);
                if (replacementCondition(featurePool[newId], featuresIDs, sample))
                {
                    numberOfPossibleExchanges--;
                    *positionInDataset = featurePool[newId];
                    break;
                }
                if (numberOfPossibleExchanges == 0)
                {
                    break;
                }
            }
        }
    }
    individual.updateDataset(features);
}

inline bool MemeticFeaturesEducation::replacementCondition(const Feature& supportVectorPoolElement,
                                                           std::unordered_set<std::uint64_t>& traningIDs,
                                                           const Feature& /*sample*/)
{
    return traningIDs.emplace(supportVectorPoolElement.id).second; // @wdudzik is id of supportVector unique in chromosome dataset
}

inline std::vector<Feature> MemeticFeaturesEducation::setDifference(const std::unordered_set<uint64_t>& svPool, const std::vector<Feature>& traningDataset)
{
    std::vector<Feature> weakSamples;
    std::copy_if(traningDataset.begin(),
                 traningDataset.end(),
                 std::back_inserter(weakSamples),
                 [&svPool](const auto& dataVector)
                 {
                     return svPool.find(dataVector.id) == svPool.end();
                 });
    return weakSamples;
}
} // namespace svmComponents
