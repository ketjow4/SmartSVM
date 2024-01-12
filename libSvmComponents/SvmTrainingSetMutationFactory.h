#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "GeneticAlgorithmsConfigs.h"

namespace svmComponents
{
enum class SvmTrainingSetMutation
{
	Unknown,
	GaSvm,
	GaSvmRegression,
	EnhanceTrainingSet
};

class SvmTrainingSetMutationFactory
{
public:
	static MutationOperator<SvmTrainingSetChromosome> create(const platform::Subtree& config,
	                                                         const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                         const std::vector<unsigned int>& labelsCount,
	                                                         std::shared_ptr<ITrainingSet> enhanceTrainingSet);

private:
	const static std::unordered_map<std::string, SvmTrainingSetMutation> m_translationsSvmKernelMutation;
};
} // namespace svmComponents
