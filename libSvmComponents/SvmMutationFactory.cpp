

#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "MutationKernelParameters.h"
#include "SvmMutationFactory.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmKernelMutation> SvmMutationFactory::m_translationsSvmKernelMutation =
{
    {"ParameterMutation", SvmKernelMutation::ParameterMutation}
};

std::unique_ptr<geneticComponents::IMutationOperator<SvmKernelChromosome>> SvmMutationFactory::create(const platform::Subtree& config)
{
    auto name = config.getValue<std::string>("Mutation.Name");

    switch (platform::stringToEnum(name, m_translationsSvmKernelMutation))
    {
    case SvmKernelMutation::ParameterMutation:
    {
        auto maxMutationChangeInPercent = config.getValue<double>("Mutation.ParameterMutation.MaxMutationChangeInPercent");
        auto mutationProbability = config.getValue<double>("Mutation.ParameterMutation.Probability");
        return std::make_unique<MutationKernelParameters>(std::move(random::RandomNumberGeneratorFactory::create(config)),
                                                          platform::Percent(maxMutationChangeInPercent),
                                                          platform::Percent(mutationProbability));
    }
    default:
        throw UnknownEnumType(name, typeid(SvmKernelMutation).name());
    }
}
} // namespace svmComponents
