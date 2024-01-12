#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "SvmTrainingSetPopulationFactory.h"
#include "SvmUtils.h"
#include "GaSvmGeneration.h"
#include "GaSvmRegression.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmTrainingSetGeneration> SvmTrainingSetPopulationFactory::m_translationsGeneration =
{
	{"Random", SvmTrainingSetGeneration::Random},
	{"GaSvmRegression", SvmTrainingSetGeneration::GaSvmRegression},
	{"EnhanceTrainingSet", SvmTrainingSetGeneration::EnhanceTrainingSet},
};

unsigned int getNumberOfClassExamples2(unsigned int numberOfClassExamples, std::vector<unsigned int> labelsCount)
{
	auto minorityClassExamplesNumber = static_cast<unsigned int>(*std::min_element(labelsCount.begin(), labelsCount.end()));
	if (minorityClassExamplesNumber < numberOfClassExamples)
		return minorityClassExamplesNumber;
	return numberOfClassExamples;
}

PopulationGeneration<SvmTrainingSetChromosome> SvmTrainingSetPopulationFactory::create(const platform::Subtree& config,
                                                                                       const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                                       const std::vector<unsigned int>& labelsCount,
                                                                                       std::shared_ptr<ITrainingSet> enhanceTrainingSet)
{
	auto name = config.getValue<std::string>("Generation.Name");

	switch (platform::stringToEnum(name, m_translationsGeneration))
	{
	case SvmTrainingSetGeneration::Random:
	{
		auto numberOfClassExamples = getNumberOfClassExamples2(config.getValue<unsigned int>("NumberOfClassExamples"), labelsCount); //fix for small highly imbalanced datasets
		//auto numberOfClassExamples = config.getValue<unsigned int>("NumberOfClassExamples");
		return std::make_unique<GaSvmGeneration>(trainingSet,
		                                         std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                         numberOfClassExamples,
		                                         labelsCount);
	}
	case SvmTrainingSetGeneration::GaSvmRegression:
	{
		auto numberOfClassExamples = config.getValue<unsigned int>("NumberOfClassExamples");
		return std::make_unique<GaSvmGenerationRegression>(trainingSet,
		                                                   std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                                   numberOfClassExamples,
		                                                   labelsCount);
	}
	case SvmTrainingSetGeneration::EnhanceTrainingSet:
	{
		SetupAdditionalVectors* a = dynamic_cast<SetupAdditionalVectors*>(enhanceTrainingSet.get());

		auto numberOfClassExamples = config.getValue<unsigned int>("NumberOfClassExamples");
		return std::make_unique<GenerationWithAdditionalExamples>(trainingSet,
			std::move(random::RandomNumberGeneratorFactory::create(config)),
			numberOfClassExamples,
			labelsCount,
			*a);
	}
	default:
		throw UnknownEnumType(name, typeid(SvmTrainingSetGeneration).name());
	}
}
} // namespace svmComponents
