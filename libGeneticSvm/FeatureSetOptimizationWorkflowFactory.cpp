
#include "libPlatform/EnumStringConversions.h"
#include "FeatureSetOptimizationWorkflowFactory.h"
#include "FeatureSelectionWorkflow.h"
#include "SvmWorkflowConfigStruct.h"
#include "MemeticFeaturesSelection.h"

namespace genetic
{
const std::unordered_map<std::string, FeatureSetOptimizationAlgorithms> FeatureSetOptimizationWorkflowFactory::m_translations =
{
    {"FeatureSetSelection", FeatureSetOptimizationAlgorithms::FeatureSetSelection},
    {"MemeticFeatureSetSelection", FeatureSetOptimizationAlgorithms::MemeticFeatureSetSelection}
};

FeatureSetOptimizationWorkflow FeatureSetOptimizationWorkflowFactory::create(const platform::Subtree& config,
                                                                             IDatasetLoader& loadingWorkflow,
                                                                             const std::string& node)
{

	
    auto name = config.getValue<std::string>(node + ".FeatureSetOptimization.Name");

    switch (platform::stringToEnum(name, m_translations))
    {
   /* case FeatureSetOptimizationAlgorithms::FeatureSetSelection:
    {
        return std::make_unique<FeatureSelectionWorkflow>(SvmWokrflowConfiguration(config),
                                                          svmComponents::GeneticFeatureSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                                          loadingWorkflow);
    }*/
    case FeatureSetOptimizationAlgorithms::MemeticFeatureSetSelection:
    {
        return std::make_unique<MemeticFeaturesSelection>(SvmWokrflowConfiguration(config),
                                                          svmComponents::MemeticFeatureSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                                          loadingWorkflow);
    }


    default:
        throw svmComponents::UnknownEnumType(name, typeid(FeatureSetOptimizationAlgorithms).name());
    }
}
} // namespace genetic
