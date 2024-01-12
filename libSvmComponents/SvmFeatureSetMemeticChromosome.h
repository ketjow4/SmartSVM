#pragma once

#include <unordered_set>
#include "libSvmComponents/BaseSvmChromosome.h"
#include "Feature.h"

namespace svmComponents
{
class SvmFeatureSetMemeticChromosome : public BaseSvmChromosome
{
public:
    SvmFeatureSetMemeticChromosome() = default;
    explicit SvmFeatureSetMemeticChromosome(std::vector<Feature>&& traningSet);

    const std::vector<Feature>& getDataset() const;
    void updateDataset(std::vector<Feature>& traningSet);
    dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const;
    std::unordered_set<std::uint64_t> convertToSet() const;
    std::size_t size() const;
    bool operator==(const SvmFeatureSetMemeticChromosome& right) const;
   
private:
    std::vector<Feature> m_featureSet;
};
} // namespace svmComponents
