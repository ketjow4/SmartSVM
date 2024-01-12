
#include "EducationOfTrainingSet.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"

namespace svmComponents
{
EducationOfTrainingSet::EducationOfTrainingSet(platform::Percent educationProbability,
                                               unsigned int numberOfClasses,
                                               std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
                                               std::unique_ptr<ISupportVectorSelection> supportVectorSelection)
    : m_educationProbability(educationProbability)
    , m_numberOfClasses(numberOfClasses)
    , m_rngEngine(std::move(randomNumberGenerator))
    , m_supportVectorSelection(std::move(supportVectorSelection))
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
    if(m_supportVectorSelection == nullptr)
    {
        throw MemberNullPointer("m_supportVectorSelection");
    }
}

void EducationOfTrainingSet::educatePopulation(geneticComponents::Population<SvmTrainingSetChromosome>& population,
                                               const std::vector<DatasetVector>& supportVectorPool,
                                               const std::vector<geneticComponents::Parents<SvmTrainingSetChromosome>>& parents,
                                               const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
    auto parentIterator = parents.begin();
    for (auto& individual : population)
    {
        educate(individual, *parentIterator, supportVectorPool, trainingSet);
        ++parentIterator;
    }
}

bool EducationOfTrainingSet::replacementCondition(const DatasetVector& supportVectorPoolElement,
                                                  std::unordered_set<std::uint64_t>& traningIDs,
                                                  const DatasetVector& sample)
{
    return sample.classValue == supportVectorPoolElement.classValue && // @wdudzik class value match
            traningIDs.emplace(supportVectorPoolElement.id).second; // @wdudzik is id of supportVector unique in chromosome dataset
}

void EducationOfTrainingSet::educate(SvmTrainingSetChromosome& individual,
                                     const geneticComponents::Parents<SvmTrainingSetChromosome>& parents,
                                     const std::vector<DatasetVector>& supportVectorPool,
                                     const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
    auto dataset = individual.getDataset();
    auto parentsSupportVectors = findSupportVectors(parents, trainingSet);
    auto weakSamples = setDifference(parentsSupportVectors, dataset);
    auto traningIDs = individual.convertToSet();

    //@wdudzik SupportVectorPool - traningIDs; possible replecement for weakSamples
    auto result = setDifference(traningIDs, supportVectorPool);

    if (result.empty())
    {
        return;
    }

    auto numberOfPossibleExchanges = svmUtils::countLabels(m_numberOfClasses, result);
    std::bernoulli_distribution education(m_educationProbability.m_percentValue);
    const auto endIndex = static_cast<int>(supportVectorPool.size() - 1);
    std::uniform_int_distribution<int> supportVectorPoolID(0, endIndex);

    for (const auto& sample : weakSamples)
    {
        if (m_rngEngine->getRandom(education))
        {
            auto positionInDataset = std::find(dataset.begin(), dataset.end(), sample);

            //@wdudzik try to replace with one form supportVectorPool
            while (true)
            {
                auto newId = m_rngEngine->getRandom(supportVectorPoolID);
                if (replacementCondition(supportVectorPool[newId], traningIDs, sample))
                {
                    numberOfPossibleExchanges[static_cast<int>(sample.classValue)]--;
                    *positionInDataset = supportVectorPool[newId];
                    break;
                }
                if (numberOfPossibleExchanges[static_cast<int>(sample.classValue)] == 0)
                {
                    break;
                }
            }
        }
    }
    individual.updateDataset(dataset);
}

std::vector<DatasetVector> EducationOfTrainingSet::setDifference(const std::unordered_set<uint64_t>& svPool,
                                                                 const std::vector<DatasetVector>& traningDataset)
{
    std::vector<DatasetVector> weakSamples;
    std::copy_if(traningDataset.begin(),
                 traningDataset.end(),
                 std::back_inserter(weakSamples),
                 [&svPool](const auto& dataVector)
         {
             return svPool.find(dataVector.id) == svPool.end();
         });
    return weakSamples;
}

std::unordered_set<uint64_t> EducationOfTrainingSet::findSupportVectors(const geneticComponents::Parents<SvmTrainingSetChromosome>& parents,
                                                                        const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
    m_supportVectorSelection->addSupportVectors(parents.first, trainingSet);
    m_supportVectorSelection->addSupportVectors(parents.second, trainingSet);

    return m_supportVectorSelection->getSupportVectorIds();
}
} // namespace svmComponents
