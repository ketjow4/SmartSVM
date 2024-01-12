
#pragma once

#include <unordered_set>
#include "libSvmComponents/BaseSvmChromosome.h"

namespace svmComponents
{
struct DatasetVector
{
    std::uint64_t id;
    float classValue;

    DatasetVector() : id(0)
                    , classValue(0)
    {
    }

    DatasetVector(std::uint64_t id, float classValue) : id(id)
                                                      , classValue(classValue)
    {
    }

    bool operator==(const DatasetVector& other) const
    {
        return id == other.id;
    }

    bool operator<(const DatasetVector& other) const
    {
        return id < other.id;
    }
};

class SvmTrainingSetChromosome : public BaseSvmChromosome
{
public:
    SvmTrainingSetChromosome();
    explicit SvmTrainingSetChromosome(std::vector<DatasetVector>&& traningSet);
	explicit SvmTrainingSetChromosome(std::vector<DatasetVector>&& traningSet, std::vector<DatasetVector>&& validationSet);

    const std::vector<DatasetVector>& getDataset() const;
    void updateDataset(std::vector<DatasetVector>& traningSet);
    dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const;
    std::unordered_set<std::uint64_t> convertToSet() const;
    std::size_t size() const;
    bool operator==(const SvmTrainingSetChromosome& right) const;

    const std::vector<DatasetVector>& getValidationDataset() const
    {
		return m_validationSet;
    }

    void updateValidationDataset(const std::vector<DatasetVector>& validationSet)
	{
		m_validationSet = validationSet;
	}

	dataset::Dataset<std::vector<float>, float> convertValidationChromosome(const dataset::Dataset<std::vector<float>, float>& validationSet) const;
	std::unordered_set<std::uint64_t> convertToValidationSet() const;
	
	std::size_t sizeValidation() const
	{
		return m_validationSet.size();
	}

private:
    std::vector<DatasetVector> m_trainingSet;
    std::vector<DatasetVector> m_validationSet;
};

inline const std::vector<DatasetVector>& SvmTrainingSetChromosome::getDataset() const
{
    return m_trainingSet;
}

inline void SvmTrainingSetChromosome::updateDataset(std::vector<DatasetVector>& traningSet)
{
    m_trainingSet = traningSet;
}

inline std::size_t SvmTrainingSetChromosome::size() const
{
    return m_trainingSet.size();
}
} // namespace svmComponents
