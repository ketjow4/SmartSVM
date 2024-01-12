
#pragma once

#include <unordered_set>
#include <gsl/span>
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "ISupportVectorSelection.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class SupportVectorPool : public ISupportVectorSelection
{
public:
    SupportVectorPool() = default;
    void updateSupportVectorPool(const geneticComponents::Population<SvmTrainingSetChromosome>& population,
                                 const dataset::Dataset<std::vector<float>, float>& trainingSet);
    void addSupportVectors(const SvmTrainingSetChromosome& chromosome,
                           const dataset::Dataset<std::vector<float>, float>& trainingSet) override;

    const std::vector<DatasetVector>& getSupportVectorPool() const override;
    const std::unordered_set<uint64_t>& getSupportVectorIds() const override;

private:
    static unsigned int findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
                                                    gsl::span<const float> supportVector);

    std::vector<DatasetVector> m_supportVectorPool;
    std::unordered_set<uint64_t> m_supportVectorIds;
};

inline const std::vector<DatasetVector>& SupportVectorPool::getSupportVectorPool() const
{
    return m_supportVectorPool;
}

inline const std::unordered_set<uint64_t>& SupportVectorPool::getSupportVectorIds() const
{
    return m_supportVectorIds;
}
} // namespace svmComponents
