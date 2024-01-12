
#include "SvmTrainingSetChromosome.h"

namespace svmComponents
{
SvmTrainingSetChromosome::SvmTrainingSetChromosome()
{
}

SvmTrainingSetChromosome::SvmTrainingSetChromosome(std::vector<DatasetVector>&& traningSet)
    : m_trainingSet(std::move(traningSet))
	, m_validationSet(std::vector<DatasetVector>()) //empty on purpose
{
}

SvmTrainingSetChromosome::SvmTrainingSetChromosome(std::vector<DatasetVector>&& traningSet, std::vector<DatasetVector>&& validationSet)
	: m_trainingSet(std::move(traningSet))
	, m_validationSet(std::move(validationSet))
{
}

dataset::Dataset<std::vector<float>, float> SvmTrainingSetChromosome::convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
{
    dataset::Dataset<std::vector<float>, float> dataset;
    auto samples = trainingDataSet.getSamples();

	dataset.reserve(m_trainingSet.size());
	
    for (auto& vector : m_trainingSet)
    {
        dataset.addSample(std::move(samples[static_cast<std::size_t>(vector.id)]), vector.classValue);
    }
    return dataset;
}


std::unordered_set<std::uint64_t> SvmTrainingSetChromosome::convertToSet() const
{
    std::unordered_set<std::uint64_t> trainingSet;
    for (const auto& element : m_trainingSet)
    {
        trainingSet.emplace(element.id);
    }
    return trainingSet;
}

bool SvmTrainingSetChromosome::operator==(const SvmTrainingSetChromosome& right) const
{
    return m_trainingSet == right.getDataset(); //TODO think if this need update for checking validationSet
}

dataset::Dataset<std::vector<float>, float> SvmTrainingSetChromosome::convertValidationChromosome(
	const dataset::Dataset<std::vector<float>, float>& validationSet) const
{
	dataset::Dataset<std::vector<float>, float> dataset;
	auto samples = validationSet.getSamples();

	for (auto& vector : m_validationSet)
	{
		dataset.addSample(std::move(samples[static_cast<std::size_t>(vector.id)]), vector.classValue);
	}
	return dataset;
}

std::unordered_set<std::uint64_t> SvmTrainingSetChromosome::convertToValidationSet() const
{
	std::unordered_set<std::uint64_t> trainingSet;
	for (const auto& element : m_validationSet)
	{
		trainingSet.emplace(element.id);
	}
	return trainingSet;
}
} // namespace svmComponents