
#include "SvmComponentsExceptions.h"
#include "FoldCreator.h"
#include <random>

namespace svmComponents
{
FoldCreator::FoldCreator(unsigned int numberOfFolds, const dataset::Dataset<std::vector<float>, float>& dataset)
    : m_numberOfFolds(numberOfFolds)
    , m_dataset(dataset)
{
}

std::pair<dataset::Dataset<std::vector<float>, float>, dataset::Dataset<std::vector<float>, float>> FoldCreator::getFold(unsigned int foldNumber)
{
    if(foldNumber >= m_numberOfFolds)
    {
        const auto maxFoldNumber = m_numberOfFolds - 1;
        const auto minFoldNumber = 0u;
        throw ValueNotInRange("foldNumber", foldNumber, minFoldNumber, maxFoldNumber);
    }

    const auto sizeOfFold = m_dataset.size() / m_numberOfFolds;
    auto samples = m_dataset.getSamples();
    auto target = m_dataset.getLabels();

    dataset::Dataset<std::vector<float>, float> training, test;

	//add radomness to folds
	std::vector<int> indexes;
	indexes.reserve(samples.size());
	for (auto i = 0u; i < samples.size(); ++i)
		indexes.emplace_back(i);
    std::shuffle(indexes.begin(), indexes.end(), std::mt19937(0));


    for (auto i = 0u; i < m_dataset.size(); ++i)
    {
        if (i >= foldNumber * sizeOfFold && i < (foldNumber + 1) * sizeOfFold)
        {
            test.addSample(samples[indexes[i]], target[indexes[i]]);
        }
        else
        {
            training.addSample(samples[indexes[i]], target[indexes[i]]);
        }
    }
    return std::pair<dataset::Dataset<std::vector<float>, float>, dataset::Dataset<std::vector<float>, float>>(training, test);
}
} // namespace svmComponents
