#include "MemeticFeaturesFactories.h"
#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "MemeticFeatureMutation.h"
#include "MemeticFeatureCrossover.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmMemeticFeatureSetMutation> SvmMemeticFeatureSetMutationFactory::m_translationsSvmKernelMutation =
{
    {"GaSvm", SvmMemeticFeatureSetMutation::GaSvm}
};

MutationOperator<SvmFeatureSetMemeticChromosome> SvmMemeticFeatureSetMutationFactory::create(const platform::Subtree& config,
                                                                                             const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                                             const std::vector<unsigned int>& labelsCount)
{
    auto name = config.getValue<std::string>("Mutation.Name");

    switch (platform::stringToEnum(name, m_translationsSvmKernelMutation))
    {
    case SvmMemeticFeatureSetMutation::GaSvm:
    {
        auto exchangePercent = config.getValue<double>("Mutation.GaSvm.ExchangePercent");
        auto mutationProbability = config.getValue<double>("Mutation.GaSvm.MutationProbability");

        return std::make_unique<MemeticFeatureMutation>(std::move(my_random::RandomNumberGeneratorFactory::create(config)),
                                                        platform::Percent(exchangePercent),
                                                        platform::Percent(mutationProbability),
                                                        trainingSet,
                                                        labelsCount);
    }
    default:
        throw UnknownEnumType(name, typeid(SvmMemeticFeatureSetMutation).name());
    }
}

const std::unordered_map<std::string, SvmMemeticFeatureSetCrossover> SvmMemeticFeatureSetCrossoverFactory::m_translationsSvmKernelCrossover =
{
    {"Memetic", SvmMemeticFeatureSetCrossover::Memetic}
};

CrossoverOperator<SvmFeatureSetMemeticChromosome> SvmMemeticFeatureSetCrossoverFactory::create(const platform::Subtree& config)
{
    auto name = config.getValue<std::string>("Crossover.Name");

    switch (platform::stringToEnum(name, m_translationsSvmKernelCrossover))
    {
    case SvmMemeticFeatureSetCrossover::Memetic:
    {
        auto numberOfClasses = config.getValue<unsigned int>("NumberOfClasses");
        return std::make_unique<MemeticFeatureCrossover>(std::move(my_random::RandomNumberGeneratorFactory::create(config)),
                                                         numberOfClasses);
    }
    default:
        throw UnknownEnumType(name, typeid(SvmMemeticFeatureSetCrossover).name());
    }
}
} // namespace svmComponents
