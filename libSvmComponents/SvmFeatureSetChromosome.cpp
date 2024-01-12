
#include "SvmFeatureSetChromosome.h"

namespace svmComponents
{
SvmFeatureSetChromosome::SvmFeatureSetChromosome(std::vector<bool>&& genes)
    : m_featureSet(std::move(genes))
{
}

dataset::Dataset<std::vector<float>, float> SvmFeatureSetChromosome::convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
    const auto sum = std::accumulate(std::begin(m_featureSet), std::end(m_featureSet), 0);

    dataset::Dataset<std::vector<float>, float> dataset;

    auto samples = trainingSet.getSamples();
    const auto targets = trainingSet.getLabels();
    int j = 0;

    for (const auto& sample : samples)
    {
        std::vector<float> filteredVector;
        filteredVector.reserve(sum);

        for(auto i = 0u; i < sample.size(); ++i)
        {
            if(m_featureSet[i])
            {
                filteredVector.emplace_back(sample[i]);
            }
        }
        dataset.addSample(std::move(filteredVector), targets[j++]);
    }
    return dataset;
}

bool SvmFeatureSetChromosome::operator==(const SvmFeatureSetChromosome& right) const
{
    return m_featureSet == right.getGenes();
}
} // namespace svmComponents