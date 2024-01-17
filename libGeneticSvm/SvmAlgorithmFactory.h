
#pragma once

#include "libPlatform/Subtree.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/GridSearchWorkflowOpenCV.h"

namespace genetic
{
class IDatasetLoader;

enum class SvmAlgorithm
{
    Unknown,
    GridSearch,
    GridSearchOpenCV,  //old do not use
    GeneticKernelEvolution, //genetic algorithm kernel optimization (@wdudzik)
    GaSvm, //training set optimization
    Alga, //kernel and training set alternating optimization
	Alma,
	FSAlma,
    Memetic,
    FeatureSetSelection,
    KTF,
    FT, // not good
    TF, // not good
    SSVM,
    RandomSearch,
    RandomSearchInitPop,
    RandomSearchEvoHelp,
    CustomKernel,   //test ground, run sequential gamma
	SequentialGamma,
	MultipleGammaMASVM,
	RbfLinear,
    RbfLinearCoevolution,
    SequentialGammaFS, //Not finihed
	ImplementationTest,
	Ensemble,
    EnsembleTree,
    SESVM_Corrected,
	BigSetsEnsemble,
};

class SvmAlgorithmFactory
{
public:
    static std::unique_ptr<genetic::ISvmAlgorithm> createAlgorightm(
        const platform::Subtree& config,
        IDatasetLoader& loadingWorkflow);

private:
    const static std::unordered_map<std::string, SvmAlgorithm> m_translations;
};
} // namespace genetic
