#include "libPlatform/EnumStringConversions.h"
#include "libGeneticComponents/CrossoverSelectionFactory.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "SvmTrainingSetCrossoverFactory.h"
#include "SvmUtils.h"
#include "GaSvmCrossover.h"
#include "MemeticCrossover.h"
#include "GaSvmRegression.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmTrainingSetCrossover> SvmTrainingSetCrossoverFactory::m_translations =
{
	{"GaSvm", SvmTrainingSetCrossover::GaSvm},
	{"Memetic", SvmTrainingSetCrossover::Memetic},
	{"GaSvmRegression", SvmTrainingSetCrossover::GaSvmRegression},
	{"EnhanceTrainingSet", SvmTrainingSetCrossover::EnhanceTrainingSet},
};

CrossoverOperator<SvmTrainingSetChromosome> SvmTrainingSetCrossoverFactory::create(const platform::Subtree& config, std::shared_ptr<ITrainingSet> enhanceTrainingSet)
{
	auto name = config.getValue<std::string>("Crossover.Name");

	switch (platform::stringToEnum(name, m_translations))
	{
	case SvmTrainingSetCrossover::GaSvm:
	{
		auto numberOfClasses = config.getValue<unsigned int>("NumberOfClasses");
		return std::make_unique<GaSvmCrossover>(std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                        numberOfClasses);
	}
	case SvmTrainingSetCrossover::Memetic:
	{
		auto numberOfClasses = config.getValue<unsigned int>("NumberOfClasses");
		return std::make_unique<MemeticCrossover>(std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                          numberOfClasses);
	}
	case SvmTrainingSetCrossover::GaSvmRegression:
	{
		auto numberOfClasses = config.getValue<unsigned int>("NumberOfClasses");
		return std::make_unique<GaSvmCrossoverRegression>(std::move(random::RandomNumberGeneratorFactory::create(config)),
		                                                  numberOfClasses);
	}
	case SvmTrainingSetCrossover::EnhanceTrainingSet:
	{
		auto numberOfClasses = config.getValue<unsigned int>("NumberOfClasses");
		return std::make_unique<CrossoverWithAdditionalExamples>(std::move(random::RandomNumberGeneratorFactory::create(config)),
			numberOfClasses,
			*dynamic_cast<SetupAdditionalVectors*>(enhanceTrainingSet.get()));
	}
	default:
		throw UnknownEnumType(name, typeid(SvmTrainingSetCrossover).name());
	}
}
} // namespace svmComponents
