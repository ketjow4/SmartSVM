#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "GeneticAlgorithmsConfigs.h"

namespace svmComponents
{
enum class SvmTrainingSetGeneration
{
	Unknown,
	Random,
	GaSvmRegression,
	EnhanceTrainingSet,
};

unsigned int getNumberOfClassExamples2(unsigned int numberOfClassExamples, std::vector<unsigned int> labelsCount);

class SvmTrainingSetPopulationFactory
{
public:
	static PopulationGeneration<SvmTrainingSetChromosome> create(const platform::Subtree& config,
	                                                             const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                             const std::vector<unsigned int>& labelsCount,
	                                                             std::shared_ptr<ITrainingSet> enhanceTrainingSet);

private:
	const static std::unordered_map<std::string, SvmTrainingSetGeneration> m_translationsGeneration;
};
} // namespace svmComponents
