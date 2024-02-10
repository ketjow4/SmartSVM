
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
    //GridSearchOpenCV,  //old do not use, not supported
    GeneticKernelEvolution, //genetic algorithm kernel optimization (@wdudzik)
    GaSvm, //training set optimization
    Alga, //kernel and training set alternating optimization
	Alma, //as above but with memetic algorithm for training set optimization
	FSAlma,
    Memetic,
    FeatureSetSelection, //Not to be used alone
    KTF, // Gecco 2019 called ESVM in paper, better to use SESVM
    FT, // not good
    TF, // not good
    SESVM,
    RandomSearch,
    RandomSearchInitPop,
    RandomSearchEvoHelp,
    CustomKernel,   //test ground, run sequential gamma
	SequentialGamma,  // ARBF-SVM
	MultipleGammaMASVM, // only for comparison
	RbfLinear,    //DASVM basic
    RbfLinearCoevolution, // DASVM with coevolution schema
    SequentialGammaFS, //Not finihed
	ImplementationTest,
	Ensemble,     //Based on ALMA algorithm, simple ensemble of many SVMs
    EnsembleTree, // CE-SVM - it is not tree ;) 
    SESVM_Corrected,  // This fix for parent selection, did not affect the results
	BigSetsEnsemble,   // ECE-SVM
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
