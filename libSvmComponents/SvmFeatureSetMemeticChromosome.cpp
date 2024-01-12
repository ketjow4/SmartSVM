#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
SvmFeatureSetMemeticChromosome::SvmFeatureSetMemeticChromosome(std::vector<Feature>&& traningSet)
    : m_featureSet(std::move(traningSet))
{
}

//#pragma optimize("", off)

dataset::Dataset<std::vector<float>, float> SvmFeatureSetMemeticChromosome::convertChromosome(
    const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
{
    //const auto sum = std::accumulate(std::begin(m_featureSet), std::end(m_featureSet), 0);

    
    std::vector<std::vector<float>> filteredSamples;
    filteredSamples.reserve(trainingDataSet.size());

    auto samples = trainingDataSet.getSamples();
    const auto targets = trainingDataSet.getLabels();
    //int j = 0;

    //auto set = convertToSet();

    for (const auto& sample : samples)
    {
        std::vector<float> filteredVector;
        filteredVector.reserve(m_featureSet.size());

        for (auto i = 0u; i < m_featureSet.size(); ++i)
        {
            filteredVector.emplace_back(sample[m_featureSet[i].id]);
            /*if (set.find(i) != set.end())
            {
                filteredVector.emplace_back(sample[i]);
            }*/
        }
        filteredSamples.emplace_back(std::move(filteredVector));
    }

    return dataset::Dataset<std::vector<float>, float>(std::move(filteredSamples), std::move(targets));
    //dataset::Dataset<std::vector<float>, float> dataset;
    //return dataset;
}

//#pragma optimize("", on)

std::unordered_set<std::uint64_t> SvmFeatureSetMemeticChromosome::convertToSet() const
{
    std::unordered_set<std::uint64_t> trainingSet;
    for (const auto& element : m_featureSet)
    {
        trainingSet.emplace(element.id);
    }
    return trainingSet;
}

bool SvmFeatureSetMemeticChromosome::operator==(const SvmFeatureSetMemeticChromosome& right) const
{
    return m_featureSet == right.getDataset();
}

const std::vector<Feature>& SvmFeatureSetMemeticChromosome::getDataset() const
{
    return m_featureSet;
}

void SvmFeatureSetMemeticChromosome::updateDataset(std::vector<Feature>& traningSet)
{
    m_featureSet = traningSet;
}

std::size_t SvmFeatureSetMemeticChromosome::size() const
{
    return m_featureSet.size();
}
} // namespace svmComponents
