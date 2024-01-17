
#include "libPlatform/EnumStringConversions.h"
#include "TrainingSetOptimizationWorkflowFactory.h"
#include "SvmWorkflowConfigStruct.h"
#include "MemeticTrainingSetWorkflow.h"
#include "GaSvmWorkflow.h"

namespace genetic
{
const std::unordered_map<std::string, TrainingSetOptimizationAlgorithms> TrainingSetOptimizationWorkflowFactory::m_translations =
{
    {"GaSvm", TrainingSetOptimizationAlgorithms::GaSvm},
    {"MemeticTrainingSetSelection", TrainingSetOptimizationAlgorithms::Memetic}
};

TrainingSetOptimizationWorkflow TrainingSetOptimizationWorkflowFactory::create(const platform::Subtree& config,
                                                                               IDatasetLoader& loadingWorkflow,
                                                                               const std::string& node)
{
    auto name = config.getValue<std::string>(node + ".TrainingSetOptimization.Name");

    switch (platform::stringToEnum(name, m_translations))
    {
    case TrainingSetOptimizationAlgorithms::GaSvm:
    {
        return std::make_unique<GaSvmWorkflow>(SvmWokrflowConfiguration(config),
                                               svmComponents::GeneticTrainingSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                               loadingWorkflow);
    }
    case TrainingSetOptimizationAlgorithms::Memetic:
    {
        return std::make_unique<MemeticTraningSetWorkflow>(SvmWokrflowConfiguration(config),
                                                           svmComponents::MemeticTrainingSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                                           loadingWorkflow,
														   config);
    }
    default:
        throw svmComponents::UnknownEnumType(name, typeid(TrainingSetOptimizationAlgorithms).name());
    }
}
} // namespace genetic
