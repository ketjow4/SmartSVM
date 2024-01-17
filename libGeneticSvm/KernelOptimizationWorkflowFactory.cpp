
#include "libPlatform/EnumStringConversions.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "KernelOptimizationWorkflowFactory.h"
#include "SvmWorkflowConfigStruct.h"

namespace genetic
{
const std::unordered_map<std::string, KernelOptimizationAlgorithms> KernelOptimizationWorkflowFactory::m_translations =
{
    {"GeneticKernelEvolution", KernelOptimizationAlgorithms::GeneticKernelEvolution}
};

KernelOptimizationWorkflow KernelOptimizationWorkflowFactory::create(const platform::Subtree& config,
                                                                     IDatasetLoader& loadingWorkflow,
                                                                     const std::string& node)
{
    auto name = config.getValue<std::string>(node + ".KernelOptimization.Name");

    switch (platform::stringToEnum(name, m_translations))
    {
    case KernelOptimizationAlgorithms::GeneticKernelEvolution:
    {
	    return std::make_unique<GeneticKernelEvolutionWorkflow>(SvmWokrflowConfiguration(config),
	                                                            svmComponents::GeneticKernelEvolutionConfiguration(config),
	                                                            loadingWorkflow,
	                                                            config);
    }
    default:
        throw svmComponents::UnknownEnumType(name, typeid(KernelOptimizationAlgorithms).name());
    }
}
} // namespace genetic
