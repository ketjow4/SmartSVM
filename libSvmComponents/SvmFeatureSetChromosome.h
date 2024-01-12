
#pragma once
#include "LibGeneticComponents/BinaryChromosome.h"
#include "libSvmComponents/BaseSvmChromosome.h"
#include "libDataset/Dataset.h"

namespace svmComponents
{
class SvmFeatureSetChromosome final : public geneticComponents::BinaryChromosome, public BaseSvmChromosome
{
public:
    SvmFeatureSetChromosome() = default;
    explicit SvmFeatureSetChromosome(std::vector<bool>&& genes);

    const std::vector<bool>& getGenes() const override;
    void updateGenes(const std::vector<bool>& genes) override;

    dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

    bool operator==(const SvmFeatureSetChromosome& right) const;

private:
    std::vector<bool> m_featureSet;
};

inline const std::vector<bool>& SvmFeatureSetChromosome::getGenes() const
{
    return m_featureSet;
}

inline void SvmFeatureSetChromosome::updateGenes(const std::vector<bool>& genes)
{
    m_featureSet = genes;
}
} // namespace svmComponents