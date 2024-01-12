#pragma once

#include <optional>
#include <unordered_set>
#include "libSvmComponents/BaseSvmChromosome.h"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
struct Gene
{
    std::uint64_t id;
    float classValue;
    double gamma;
    //std::vector<double> parameters;
    //KernelType kernelType;

    Gene()
        : id(0)
        , classValue(0)
    {
    }

    Gene(std::uint64_t id, float classValue, double gamma)
        : id(id)
        , classValue(classValue)
        , gamma(gamma)
    {
    }

    bool operator==(const Gene& other) const
    {
        return id == other.id;
    }

    bool operator<(const Gene& other) const
    {
        return id < other.id;
    }
};



class SvmCustomKernelChromosome : public BaseSvmChromosome
{
public:
    SvmCustomKernelChromosome() {}
    explicit SvmCustomKernelChromosome(std::vector<Gene>&& traningSet, double c);
    explicit SvmCustomKernelChromosome(std::vector<Gene>&& traningSet, double c, const std::vector<Feature>& features);

    const std::vector<Gene>& getDataset() const;
    std::vector<double> getGammas() const;
    void updateDataset(std::vector<Gene>& traningSet);
    virtual dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const;
    std::unordered_set<std::uint64_t> convertToSet() const;
    std::size_t size() const;
    bool operator==(const SvmCustomKernelChromosome& right) const;

    double getC() const  { return C; }

    bool hasFeatures() const { return m_features.has_value(); }
    const std::vector<Feature>& getFeatures() const { return m_features.value(); }
	
private:
    std::optional<std::vector<Feature>> m_features;
    std::vector<Gene> m_trainingSet;
    double C;
};


inline SvmCustomKernelChromosome::SvmCustomKernelChromosome(std::vector<Gene>&& traningSet, double c)
    : m_trainingSet(std::move(traningSet))
    , C(c)
{
}

inline SvmCustomKernelChromosome::SvmCustomKernelChromosome(std::vector<Gene>&& traningSet, double c, const std::vector<Feature>& features)
    : m_trainingSet(std::move(traningSet))
    , C(c)
	, m_features(features)
{
}

inline const std::vector<Gene>& SvmCustomKernelChromosome::getDataset() const
{
    return m_trainingSet;
}

inline std::vector<double> SvmCustomKernelChromosome::getGammas() const
{
    std::vector<double> gammas;
    for (auto& vector : m_trainingSet)
    {
        gammas.emplace_back(vector.gamma);
    }
    return gammas;
}

inline void SvmCustomKernelChromosome::updateDataset(std::vector<Gene>& traningSet)
{
    m_trainingSet = traningSet;
}

inline dataset::Dataset<std::vector<float>, float> SvmCustomKernelChromosome::convertChromosome(
    const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
{
    dataset::Dataset<std::vector<float>, float> dataset;
    auto samples = trainingDataSet.getSamples();

    for (auto& vector : m_trainingSet)
    {
        dataset.addSample(std::move(samples[static_cast<std::size_t>(vector.id)]), vector.classValue);
    }

	if(m_features.has_value())
	{
        std::vector<Feature> temp_f;
        std::copy(m_features.value().begin(), m_features.value().end(), std::back_inserter(temp_f));
        SvmFeatureSetMemeticChromosome temp(std::move(temp_f));
        dataset = temp.convertChromosome(dataset);
	}
	
    return dataset;
}

inline std::unordered_set<std::uint64_t> SvmCustomKernelChromosome::convertToSet() const
{
    std::unordered_set<std::uint64_t> trainingSet;
    for (const auto& element : m_trainingSet)
    {
        trainingSet.emplace(element.id);
    }
    return trainingSet;
}

inline std::size_t SvmCustomKernelChromosome::size() const
{
    return m_trainingSet.size();
}

inline bool SvmCustomKernelChromosome::operator==(const SvmCustomKernelChromosome& right) const
{
    return m_trainingSet == right.getDataset();
}
} // namespace svmComponents

namespace std
{
	template<>
	struct hash<svmComponents::Gene>
	{
		size_t
			operator()(const svmComponents::Gene& obj) const
		{
			return std::hash<std::uint64_t>()(obj.id);
		}
	};
}
