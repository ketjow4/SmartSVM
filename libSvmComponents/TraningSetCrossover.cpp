
#include "TraningSetCrossover.h"
#include "SvmUtils.h"
#include "libPlatform/Percent.h"

namespace svmComponents
{
TrainingSetCrossover::TrainingSetCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                           unsigned numberOfClasses)
    : m_rngEngine(std::move(rngEngine))
    , m_numberOfClasses(numberOfClasses)
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

void TrainingSetCrossover::tryToInsert(const std::vector<DatasetVector>::const_iterator& parentAIt,
                                       const std::vector<DatasetVector>::const_iterator& parentBIt,
                                       std::unordered_set<uint64_t>& childSet,
                                       std::vector<unsigned>& classes,
                                       unsigned classExamples,
                                       std::vector<DatasetVector>& child) const
{
    if (classes[static_cast<int>(parentAIt->classValue)] < classExamples && childSet.insert(parentAIt->id).second)
    {
        ++classes[static_cast<int>(parentAIt->classValue)];
        child.emplace_back(*parentAIt);
    }
    else if (classes[static_cast<int>(parentBIt->classValue)] < classExamples && childSet.insert(parentBIt->id).second)
    {
        ++classes[static_cast<int>(parentBIt->classValue)];
        child.emplace_back(*parentBIt);
    }
}

std::vector<DatasetVector> TrainingSetCrossover::crossoverInternals(const chromosomeType& parentA,
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
    int finalBreak = 0;
    while (true && finalBreak < 100000)
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

        if (std::all_of(classes.begin(), classes.end(), [classExamples](const auto classCount)
                {
                    return classCount == classExamples;
                }))
        {
            break;
        }


        if(++parentBIt == parentB.getDataset().end())
        {
            parentBIt = parentB.getDataset().begin();
        }
        if (++parentAIt == parentA.getDataset().end())
        {
            parentAIt = parentA.getDataset().begin();
        }
        finalBreak++;
    }
    return child;
}
} // namespace svmComponents
