#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "MutationKernelParameters.h"
#include "SvmMutationFactory.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "SvmTrainingSetMutationFactory.h"
#include "GaSvmMutation.h"
#include "GeneticAlgorithmsConfigs.h"
#include "GaSvmRegression.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmTrainingSetMutation> SvmTrainingSetMutationFactory::m_translationsSvmKernelMutation =
{
	{"GaSvm", SvmTrainingSetMutation::GaSvm},
	{"GaSvmRegression", SvmTrainingSetMutation::GaSvmRegression},
	{"EnhanceTrainingSet", SvmTrainingSetMutation::EnhanceTrainingSet}
};

MutationOperator<SvmTrainingSetChromosome> SvmTrainingSetMutationFactory::create(const platform::Subtree& config,
                                                                                 const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                                 const std::vector<unsigned int>& labelsCount,
                                                                                 std::shared_ptr<ITrainingSet> enhanceTrainingSet)
{
	auto name = config.getValue<std::string>("Mutation.Name");

	switch (platform::stringToEnum(name, m_translationsSvmKernelMutation))
	{
	case SvmTrainingSetMutation::GaSvm:
	{
		auto exchangePercent = config.getValue<double>("Mutation.GaSvm.ExchangePercent");
		auto mutationProbability = config.getValue<double>("Mutation.GaSvm.MutationProbability");

		return std::make_unique<GaSvmMutation>(std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                       platform::Percent(exchangePercent),
		                                       platform::Percent(mutationProbability),
		                                       trainingSet,
		                                       labelsCount);
	}
	case SvmTrainingSetMutation::GaSvmRegression:
	{
		auto exchangePercent = config.getValue<double>("Mutation.GaSvm.ExchangePercent");
		auto mutationProbability = config.getValue<double>("Mutation.GaSvm.MutationProbability");

		return std::make_unique<GaSvmMutationRegression>(std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                                 platform::Percent(exchangePercent),
		                                                 platform::Percent(mutationProbability),
		                                                 trainingSet,
		                                                 labelsCount);
	}
	case SvmTrainingSetMutation::EnhanceTrainingSet:
	{

		return std::make_unique<MutationWithAdditionalExamples>(*dynamic_cast<SetupAdditionalVectors*>(enhanceTrainingSet.get()));
	}
	default:
		throw UnknownEnumType(name, typeid(SvmTrainingSetMutation).name());
	}
}
} // namespace svmComponents
