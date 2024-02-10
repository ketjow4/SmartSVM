
#include <unordered_map>
#include "SvmWorkflowConfigStruct.h"
#include "SvmExceptions.h"
#include "SvmAlgorithmFactory.h"
#include "GeneticKernelEvolutionWorkflow.h"
#include "GaSvmWorkflow.h"
#include "CombinedAlgorithmsConfig.h"
#include "AlgaWorkflow.h"
#include "BigSetsEnsemble.h"
#include "MemeticTrainingSetWorkflow.h"
#include "FeatureSelectionWorkflow.h"
#include "KTFGeneticWorkflow.h"
#include "GridSearchWorkflow.h"
#include "TFGeneticWorkflow.h"
#include "FTGeneticWorkflow.h"
#include "libPlatform/EnumStringConversions.h"
#include "SimultaneousWorkflow.h"
#include "RandomSearchWorkflow.h"
#include "CustomKernelWorkflow.h"
#include "EnsembleTreeWorkflow.h"
#include "ImplemneationTestWorkflow.h"
#include "EnsembleWorkflow.h"
#include "SequentialGammaWorkflow.h"
#include "MultipleGammaMASVM.h"
#include "libPlatform/loguru.hpp"
#include "RbfLinearCoevolutionWorkflow.h"
#include "RbfLinearWorkflow.h"
#include "SequentialGammaWorkflowWithFeatureSelection.h"
#include "SimultaneousWorkflowCorrected.h"

namespace genetic
{
using namespace svmComponents;

const std::unordered_map<std::string, SvmAlgorithm> SvmAlgorithmFactory::m_translations =
{
    { "GridSearch", SvmAlgorithm::GridSearch },
    { "GridSearchNoFS", SvmAlgorithm::GridSearch },
    //{ "GridSearchOpenCV", SvmAlgorithm::GridSearchOpenCV },
    { "Alga", SvmAlgorithm::Alga },
    { "Alma", SvmAlgorithm::Alma },
    { "FSAlma", SvmAlgorithm::FSAlma },
    { "GaSvm", SvmAlgorithm::GaSvm },
    { "GeneticKernelEvolution", SvmAlgorithm::GeneticKernelEvolution },
    { "MemeticTrainingSetSelection", SvmAlgorithm::Memetic },
    { "FeatureSetSelection", SvmAlgorithm::FeatureSetSelection },
    { "KTF", SvmAlgorithm::KTF },
    { "TF", SvmAlgorithm::TF },
    { "FT", SvmAlgorithm::FT },
    { "SESVM", SvmAlgorithm::SESVM},
    { "RandomSearch", SvmAlgorithm::RandomSearch },
    { "RandomSearchInitPop", SvmAlgorithm::RandomSearchInitPop},
    { "RandomSearchEvoHelp", SvmAlgorithm::RandomSearchEvoHelp },
    { "CustomKernel", SvmAlgorithm::CustomKernel },
    { "SequentialGamma", SvmAlgorithm::SequentialGamma},
    { "MultipleGammaMASVM", SvmAlgorithm::MultipleGammaMASVM},
    { "RbfLinear", SvmAlgorithm::RbfLinear},
    { "RbfLinearCoevolution", SvmAlgorithm::RbfLinearCoevolution},
    { "RbfLinearCoevolutionCEONE", SvmAlgorithm::RbfLinearCoevolution},
    { "RbfLinearCoevolutionCEMAX", SvmAlgorithm::RbfLinearCoevolution},
    { "RbfLinearCoevolutionCENOT", SvmAlgorithm::RbfLinearCoevolution},
    { "RbfLinearCoevolutionFSCEONE", SvmAlgorithm::RbfLinearCoevolution},
    { "RbfLinearCoevolutionFSCEMAX", SvmAlgorithm::RbfLinearCoevolution},
    { "RbfLinearCoevolutionFSCENOT", SvmAlgorithm::RbfLinearCoevolution},
    { "Ensemble", SvmAlgorithm::Ensemble}, 
    { "EnsembleTree", SvmAlgorithm::EnsembleTree}, 
    { "SequentialGammaFS", SvmAlgorithm::SequentialGammaFS},
    { "ImplementationTest", SvmAlgorithm::ImplementationTest},
    { "SESVM_Corrected", SvmAlgorithm::SESVM_Corrected},
    { "BigSetsEnsemble", SvmAlgorithm::BigSetsEnsemble},

};

std::unique_ptr<ISvmAlgorithm> SvmAlgorithmFactory::createAlgorightm(const platform::Subtree& config,
                                                                     IDatasetLoader& loadingWorkflow)
{
    auto algorithmName = config.getValue<std::string>("Name");

    switch (platform::stringToEnum(algorithmName, m_translations))
    {
    case SvmAlgorithm::GridSearch:
    {
        return std::make_unique<GridSearchWorkflow>(SvmWokrflowConfiguration(config),
                                                    GridSearchConfiguration(config),
                                                    loadingWorkflow);
    }
    case SvmAlgorithm::GeneticKernelEvolution:
    {
        return std::make_unique<GeneticKernelEvolutionWorkflow>(SvmWokrflowConfiguration(config),
                                                                GeneticKernelEvolutionConfiguration(config),
                                                                loadingWorkflow, config);
    }
    case SvmAlgorithm::GaSvm:
    {
        return std::make_unique<GaSvmWorkflow>(SvmWokrflowConfiguration(config),
                                               GeneticTrainingSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                               loadingWorkflow);
    }
    case SvmAlgorithm::Alga:
    {
        return std::make_unique<AlgaWorkflow>(SvmWokrflowConfiguration(config),
                                              GeneticAlternatingEvolutionConfiguration(config, loadingWorkflow));
    }
    case SvmAlgorithm::Alma:
    {
        return std::make_unique<AlgaWorkflow>(SvmWokrflowConfiguration(config),
                                              GeneticAlternatingEvolutionConfiguration(config, loadingWorkflow));
    }
    case SvmAlgorithm::FSAlma:
    {
        return std::make_unique<AlgaWorkflow>(SvmWokrflowConfiguration(config),
                                              GeneticAlternatingEvolutionConfiguration(config, loadingWorkflow));
    }
    case SvmAlgorithm::Memetic:
    {
        return std::make_unique<MemeticTraningSetWorkflow>(SvmWokrflowConfiguration(config),
                                                           MemeticTrainingSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                                           loadingWorkflow, config);
    }
    case SvmAlgorithm::FeatureSetSelection:
    {
        return std::make_unique<FeatureSelectionWorkflow>(SvmWokrflowConfiguration(config),
                                                          GeneticFeatureSetEvolutionConfiguration(config, loadingWorkflow.getTraningSet()),
                                                          loadingWorkflow);
    }
    case SvmAlgorithm::KTF:
    {
        return std::make_unique<KTFGeneticWorkflow>(SvmWokrflowConfiguration(config), KTFGeneticEvolutionConfiguration(config, loadingWorkflow));
    }
    case SvmAlgorithm::TF:
    {
        return std::make_unique<TFGeneticWorkflow>(SvmWokrflowConfiguration(config), TFGeneticEvolutionConfiguration(config, loadingWorkflow, "TF"));
    }
    case SvmAlgorithm::FT:
    {
        return std::make_unique<FTGeneticWorkflow>(SvmWokrflowConfiguration(config), TFGeneticEvolutionConfiguration(config, loadingWorkflow, "FT"));
    }
    case SvmAlgorithm::SESVM:
    {
        return std::make_unique<SimultaneousWorkflow>(SvmWokrflowConfiguration(config), SimultaneousWorkflowConfig(config, loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::RandomSearch:
    {
        //configuration has to be the same as for SSVM, most of things is hardcoded inside class
        return std::make_unique<RandomSearchWorkflow>(SvmWokrflowConfiguration(config), RandomSearchWorkflowConfig(config, loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::RandomSearchInitPop:
    {
        //configuration has to be the same as for SSVM, most of things is hardcoded inside class
        return std::make_unique<RandomSearchWithInitialPopulationsWorkflow>(SvmWokrflowConfiguration(config), RandomSearchWorkflowInitPopsConfig(config, loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::RandomSearchEvoHelp:
    {
        //configuration has to be the same as for SSVM, most of things is hardcoded inside class
        return std::make_unique<RandomSearchWithHelpFromEvolutionWorkflow>(SvmWokrflowConfiguration(config), RandomSearchWorkflowEvoHelpConfig(config, loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::CustomKernel:
    {
        LOG_F(ERROR, "This algorithm is used only for experimentation, the final version is SequentialGamma. Do not use for any serious experiments");
        //custom kernel is now only RBF with different gammas for each vector
        return std::make_unique<CustomKernelWorkflow>(SvmWokrflowConfiguration(config), CustomKernelEvolutionConfiguration(config, loadingWorkflow.getTraningSet()), loadingWorkflow);
    }
	case SvmAlgorithm::SequentialGamma:
	{
		return std::make_unique<SequentialGammaWorkflow>(SvmWokrflowConfiguration(config), SequentialGammaConfig(config, loadingWorkflow.getTraningSet()), loadingWorkflow);
	}
	case SvmAlgorithm::MultipleGammaMASVM:
	{
		return std::make_unique<MultipleGammaMASVMWorkflow>(SvmWokrflowConfiguration(config), MutlipleGammaMASVMConfig(config, loadingWorkflow.getTraningSet()), loadingWorkflow);
	}
    case SvmAlgorithm::RbfLinear:
    {
        return std::make_unique<RbfLinearWorkflow>(SvmWokrflowConfiguration(config), RbfLinearConfig(config, loadingWorkflow.getTraningSet()), loadingWorkflow);
    }
    case SvmAlgorithm::RbfLinearCoevolution:
    {
        return std::make_unique<RbfLinearCoevolutionWorkflow>(SvmWokrflowConfiguration(config), RbfLinearConfig(config, loadingWorkflow.getTraningSet()), loadingWorkflow, config);
    }
    case SvmAlgorithm::Ensemble:
    {
        return std::make_unique<EnsembleWorkflow>(SvmWokrflowConfiguration(config), GeneticAlternatingEvolutionConfiguration(config, loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::EnsembleTree:
    {
        return std::make_unique<EnsembleTreeWorkflow>(SvmWokrflowConfiguration(config), EnsembleTreeWorkflowConfig(config, loadingWorkflow), loadingWorkflow, config);
    }
    case SvmAlgorithm::SequentialGammaFS:
    {
        return std::make_unique<SequentialGammaWorkflowWithFeatureSelection>(SvmWokrflowConfiguration(config), SequentialGammaConfigWithFeatureSelection(config, loadingWorkflow.getTraningSet(), loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::ImplementationTest:
    {
        return std::make_unique<ImplementationTestnWorkflow>(SvmWokrflowConfiguration(config), GeneticKernelEvolutionConfiguration(config), loadingWorkflow);
    }
    case SvmAlgorithm::SESVM_Corrected:
    {
        return std::make_unique<SimultaneousWorkflowCorrected>(SvmWokrflowConfiguration(config), SimultaneousWorkflowConfig(config, loadingWorkflow), loadingWorkflow);
    }
    case SvmAlgorithm::BigSetsEnsemble:
    {
        return std::make_unique<BigSetsEnsemble>(SvmWokrflowConfiguration(config), EnsembleTreeWorkflowConfig(config, loadingWorkflow), loadingWorkflow, config);
    }
    default:
		LOG_F(ERROR, "Wrong algorithm name %s", algorithmName.c_str());
        throw UnsupportedAlgorithmTypeException(algorithmName);
    }
}
} // namespace genetic
