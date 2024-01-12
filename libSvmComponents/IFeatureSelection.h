#pragma once

#include <libDataset/Dataset.h>

namespace svmComponents
{
class IFeatureSelection
{
public:
    virtual ~IFeatureSelection() = default;

    virtual void addFeatures(const SvmFeatureSetMemeticChromosome& chromosome,
                             const dataset::Dataset<std::vector<float>, float>& trainingSet) = 0;

    virtual const std::vector<Feature>& getFeaturesPool() const = 0;
    virtual const std::unordered_set<uint64_t>& getFeaturesIds() const = 0;
};
} // namespace svmComponents
