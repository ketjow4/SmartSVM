#pragma once

#include "Commons.h"
#include "libSvmComponents/DataNormalization.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "libSvmComponents/SvmTraining.h"
#include "libSvmComponents/SvmKernelTraining.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "libSvmComponents/CustomKernelTraining.h"

inline void setupDatasets2D(platform::Subtree& config, std::string datafolder, std::string algorithmName, std::string subsetSize)
{
	std::string outputFolder = algorithmName  +"_recreate_experiment_test_" + subsetSize; // __k_with_regions__
	config.putValue<std::string>("Svm.OutputFolderPath", datafolder + "\\" + outputFolder + "\\");
	std::string treningName = "train.csv";
	std::string testName = "test.csv";

	config.putValue<std::string>("Svm.ValidationData", datafolder + "\\" + treningName);
	config.putValue<std::string>("Svm.TestData", datafolder + "\\" + testName);
	config.putValue<std::string>("Svm.TrainingData", datafolder + "\\" + treningName);
}

void changeConfigurationForExperiment(std::vector<std::string> dataFolders, const char* algorithmName)
{
	for (const auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto configs = testApp::getAllConfigFiles(foldFolder);
			testApp::ConfigManager configManager;
			for (auto& configPath : configs)
			{
				if(configPath.string().find(algorithmName) != configPath.string().npos)
				{				
					std::vector<int> K = {8};
					for(auto k : K)
					{
						platform::Subtree config{ configPath };

						setupDatasets2D(config, foldFolder, algorithmName, std::to_string(k));
						configManager.setK(config, k);
						config.putValue<std::string>("Svm.Metric", "AUC");
						configManager.setRandomNumberGenerators(config);

						config.putValue<double>("Svm.GaSvm.Kernel.Gamma", 1);
						config.putValue<double>("Svm.GaSvm.Kernel.C", 1);
						//config.putValue<int>("GridSearch.SubsetSize", 0); //0 means full training set, otherwise random subset of T will be selected
						//config.putValue<int>("GridSearch.SubsetRepeats", 5); //number of times subsets will be tested (does not apply when using full T)

						/*if( k == 9)
						{*/
							configManager.saveConfigToFileFolds(config, foldFolder, algorithmName, getFoldNumberFromFolder(foldFolder));
						/*}
						else
						{
							configManager.saveConfigToFileFolds(config, foldFolder, algorithmName + std::to_string(k), getFoldNumberFromFolder(foldFolder));
						}*/
						
					}
				}
			}
		}
	}
}


inline void DifferentKernelConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	auto sum = genetic::DefaultSequentialGammaConfig::getDefault();
	sum.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	//a.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Subset");
	//a.putValue<std::string>("Svm.SequentialGamma.Validation.Method", "Dummy");
	//
	auto b = genetic::DefaultSequentialGammaConfig::getDefault();
	b.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	b.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Subset");
	b.putValue<std::string>("Svm.SequentialGamma.Validation.Method", "RandomSubsetPerIteration");
	b.putValue<double>("Svm.SequentialGamma.Validation.RandomSubsetPercent", 0.5);
	b.putValue<double>("Svm.SequentialGamma.Validation.Generation", 2);

	auto b1 = genetic::DefaultSequentialGammaConfig::getDefault();
	b1.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	b1.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Subset");
	b1.putValue<std::string>("Svm.SequentialGamma.Validation.Method", "RandomSubsetPerIteration");
	b1.putValue<double>("Svm.SequentialGamma.Validation.RandomSubsetPercent", 0.1);
	b1.putValue<double>("Svm.SequentialGamma.Validation.Generation", 2);


	auto b3 = genetic::DefaultSequentialGammaConfig::getDefault();
	b3.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	b3.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Subset");
	b3.putValue<std::string>("Svm.SequentialGamma.Validation.Method", "RandomSubsetPerIteration");
	b3.putValue<double>("Svm.SequentialGamma.Validation.RandomSubsetPercent", 0.5);
	b3.putValue<double>("Svm.SequentialGamma.Validation.Generation", 3);

	auto b13 = genetic::DefaultSequentialGammaConfig::getDefault();
	b13.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	b13.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Subset");
	b13.putValue<std::string>("Svm.SequentialGamma.Validation.Method", "RandomSubsetPerIteration");
	b13.putValue<double>("Svm.SequentialGamma.Validation.RandomSubsetPercent", 0.1);
	b13.putValue<double>("Svm.SequentialGamma.Validation.Generation", 3);

	
	auto c = genetic::DefaultSequentialGammaConfig::getDefault();
	c.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	c.putValue<bool>("Svm.SequentialGamma.ShrinkOnBestOnly", true);


	auto min = genetic::DefaultSequentialGammaConfig::getDefault();
	min.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MIN");
	
	auto max = genetic::DefaultSequentialGammaConfig::getDefault();
	max.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	//d.putValue<std::string>("Svm.SequentialGamma.Validation.Name", "Subset");
	//d.putValue<std::string>("Svm.SequentialGamma.Validation.Method", "Dummy");

	auto sum_2_kernels = genetic::DefaultSequentialGammaConfig::getDefault();
	sum_2_kernels.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM_2_KERNELS");

	auto e = genetic::DefaultMultipleGammaMASVMConfig::getDefault();
	e.putValue<std::string>("Svm.MultipleGammaMASVM.KernelType", "RBF_SUM");

	auto g = genetic::DefaultMultipleGammaMASVMConfig::getDefault();
	g.putValue<std::string>("Svm.MultipleGammaMASVM.KernelType", "RBF_MIN");

	auto h = genetic::DefaultMultipleGammaMASVMConfig::getDefault();
	h.putValue<std::string>("Svm.MultipleGammaMASVM.KernelType", "RBF_MAX");

	auto i = genetic::DefaultMultipleGammaMASVMConfig::getDefault();
	i.putValue<std::string>("Svm.MultipleGammaMASVM.KernelType", "RBF_SUM_2_KERNELS");


	auto na = genetic::DefaultSequentialGammaConfig::getDefault();
	na.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);

	auto na1 = genetic::DefaultSequentialGammaConfig::getDefault();
	na1.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM");
	na1.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);

	auto na2 = genetic::DefaultSequentialGammaConfig::getDefault();
	na2.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_SUM_2_KERNELS");
	na2.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);

	auto na3_min = genetic::DefaultSequentialGammaConfig::getDefault();
	na3_min.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MIN");
	na3_min.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);

	auto na4_max = genetic::DefaultSequentialGammaConfig::getDefault();
	na4_max.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	na4_max.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);


	auto linear = genetic::DefaultSequentialGammaConfig::getDefault();
	linear.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
	linear.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");
	linear.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 1);

	auto linear2 = genetic::DefaultSequentialGammaConfig::getDefault();
	linear2.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
	linear2.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");
	linear2.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 2);
	//linear2.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);

	auto linear3 = genetic::DefaultSequentialGammaConfig::getDefault();
	linear3.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
	linear3.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");
	linear3.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 4);

	auto linear4 = genetic::DefaultSequentialGammaConfig::getDefault();
	linear4.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
	linear4.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");
	linear4.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 6);

	auto linear5 = genetic::DefaultSequentialGammaConfig::getDefault();
	linear5.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
	linear5.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");
	linear5.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 10);

	auto linear6 = genetic::DefaultSequentialGammaConfig::getDefault();
	linear6.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_LINEAR");
	linear6.putValue<std::string>("Svm.SequentialGamma.Generation.Name", "NewKernel");
	linear6.putValue<unsigned int>("Svm.SequentialGamma.NumberOfClassExamples", 20);

	auto rbfLinear = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinear.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);

	auto rbfLinearCoevolution = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution.putValue("Name", "RbfLinearCoevolution");


	auto rbfLinearCoevolution_single = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_single.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_single.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_single.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_single.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR_SINGLE");

	auto rbfLinearCoevolution_max = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_max.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_max.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_max.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_max.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR_MAX");

	auto rbfLinearCoevolution_min = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_min.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_min.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_min.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_min.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR_MIN");

	auto rbfLinearCoevolution_sum_2_kernels = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_sum_2_kernels.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_sum_2_kernels.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_sum_2_kernels.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_sum_2_kernels.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR_SUM_2_KERNELS");

	auto rbfLinearCoevolution_nosmo = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_nosmo.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_nosmo.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_nosmo.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_nosmo.putValue<bool>("Svm.RbfLinear.TrainAlpha", false);
	
	

	auto rbfLinearCoevolution50 = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution50.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution50.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution50.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution50.putValue<std::string>("Svm.RbfLinear.Validation.Name", "Subset");
	rbfLinearCoevolution50.putValue<std::string>("Svm.RbfLinear.Validation.Method", "RandomSubsetPerIteration");
	rbfLinearCoevolution50.putValue<double>("Svm.RbfLinear.Validation.RandomSubsetPercent", 0.5);

	auto rbfLinearCoevolution20 = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution20.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution20.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution20.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution20.putValue<std::string>("Svm.RbfLinear.Validation.Name", "Subset");
	rbfLinearCoevolution20.putValue<std::string>("Svm.RbfLinear.Validation.Method", "RandomSubsetPerIteration");
	rbfLinearCoevolution20.putValue<double>("Svm.RbfLinear.Validation.RandomSubsetPercent", 0.2);

	auto rbfLinearCoevolution10 = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution10.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution10.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution10.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution10.putValue<std::string>("Svm.RbfLinear.Validation.Name", "Subset");
	rbfLinearCoevolution10.putValue<std::string>("Svm.RbfLinear.Validation.Method", "RandomSubsetPerIteration");
	rbfLinearCoevolution10.putValue<double>("Svm.RbfLinear.Validation.RandomSubsetPercent", 0.1);
	

	auto ICPR_WITH_FS = genetic::DefaultSequentialGammaWithFeatureSelectionConfig::getDefault();
	if (info.numberOfFeatures < 4)
	{
		ICPR_WITH_FS.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", info.numberOfFeatures);
	}

	auto MAX_WITH_FS = genetic::DefaultSequentialGammaWithFeatureSelectionConfig::getDefault();
	if (info.numberOfFeatures < 4)
	{
		MAX_WITH_FS.putValue<unsigned int>("Svm.MemeticFeatureSetSelection.NumberOfClassExamples", info.numberOfFeatures);
	}
	MAX_WITH_FS.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");


	//

	auto na_ga = genetic::DefaultSequentialGammaConfig::getDefault();
	na_ga.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);
	na_ga.putValue<std::string>("Svm.SequentialGamma.HelperAlgorithmName", "baseGA");


	auto na_pso = genetic::DefaultSequentialGammaConfig::getDefault();
	na_pso.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);
	na_pso.putValue<std::string>("Svm.SequentialGamma.HelperAlgorithmName", "basePSO");

	auto na_de = genetic::DefaultSequentialGammaConfig::getDefault();
	na_de.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);
	na_de.putValue<std::string>("Svm.SequentialGamma.HelperAlgorithmName", "baseDE");

	auto na_mvo = genetic::DefaultSequentialGammaConfig::getDefault();
	na_mvo.putValue<bool>("Svm.SequentialGamma.TrainAlpha", false);
	na_mvo.putValue<std::string>("Svm.SequentialGamma.HelperAlgorithmName", "MVO");


	auto max_no_smaller = genetic::DefaultSequentialGammaConfig::getDefault();
	max_no_smaller.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	max_no_smaller.putValue<bool>("Svm.SequentialGamma.UseSmallerGamma", false);


	auto max_logstep3 = genetic::DefaultSequentialGammaConfig::getDefault();
	max_logstep3.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	max_logstep3.putValue<double>("Svm.SequentialGamma.GammaLogStep", 3);

	auto max_logstep5 = genetic::DefaultSequentialGammaConfig::getDefault();
	max_logstep5.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	max_logstep5.putValue<double>("Svm.SequentialGamma.GammaLogStep", 5);

	auto max_logstep7 = genetic::DefaultSequentialGammaConfig::getDefault();
	max_logstep7.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	max_logstep7.putValue<double>("Svm.SequentialGamma.GammaLogStep", 7);

	auto max_logstep5_nos = genetic::DefaultSequentialGammaConfig::getDefault();
	max_logstep5_nos.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	max_logstep5_nos.putValue<double>("Svm.SequentialGamma.GammaLogStep", 5);
	max_logstep5_nos.putValue<bool>("Svm.SequentialGamma.UseSmallerGamma", false);


	
	auto max1e2 = genetic::DefaultSequentialGammaConfig::getDefault();
	max1e2.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	max1e2.putValue<double>("Svm.SequentialGamma.StopCondition.MeanFitness.Epsilon", 5e-3);
	
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		//{"SequentialGamma_no_alpha_ga" ,na_ga},
		//{"SequentialGamma_no_alpha_pso" ,na_pso},
		//{"SequentialGamma_no_alpha_de" ,na_de},
		//{"SequentialGamma_no_alpha_mvo" ,na_mvo},

		
		//{"ICPR", genetic::DefaultSequentialGammaConfig::getDefault()},
		//{"Sum_ICPR(journal)" ,sum},
		/*{"SequentialGamma_Kernel_Gamma_Sum_ICPR_50_2" ,b},
		{"SequentialGamma_Kernel_Gamma_Sum_ICPR_10_2" ,b1},
		{"SequentialGamma_Kernel_Gamma_Sum_ICPR_50_3" ,b3},
		{"SequentialGamma_Kernel_Gamma_Sum_ICPR_10_3" ,b13},*/
		//{"SequentialGamma_shrink_on_best" ,c},
		//{"SequentialGamma_Kernel_Max", max},
		//{"SequentialGamma_Kernel_Max_standarize", max},
		//{"DASVM_Max(std,T=V,MCC_thr)", max},
		//{"DASVM_Max(norm,T=V,GS-size)", max},
		//{"DASVM_Max(std,T=V,Stop=5e3)", max1e2},
		
		//{"DASVM_Max_baseline(std,T=V)", max},
		//{"DASVM_Max_(std,T=V,NoSmallerGamma)", max_no_smaller},
		//{"DASVM_Max_(LogStep_3)", max_logstep3},
		//{"DASVM_Max_(LogStep_5)", max_logstep5},
		//{"DASVM_Max_(LogStep_7)", max_logstep7},
		//{"DASVM_Max_(LogStep_5_nos)", max_logstep5_nos},
		//{"SequentialGamma_Kernel_Max_log_step_2", max},
		//{"SequentialGamma_Kernel_Min", min},
		//{"SequentialGamma_sum_2_kernels", sum_2_kernels},

		//{"MultipleGammaMASVM_sum" ,e},
		//{"MultipleGammaMASVM_min" ,g},
		//{"MultipleGammaMASVM_max" ,h},
		//{"MultipleGammaMASVM_sum_2_kernels" ,i},

		{"SequentialGamma_no_alpha" ,na},
		/*{"SequentialGamma_no_alpha_sum" ,na1},
		{"SequentialGamma_no_alpha_sum_2_kernels" ,na2},
		{"SequentialGamma_no_alpha_min" ,na3_min},
		{"SequentialGamma_no_alpha_max" ,na4_max},*/
		
		//{"SequentialGamma_linear_1" ,linear},
		//{"SequentialGamma_linear_2" ,linear2},
		//{"SequentialGamma_linear_4" ,linear3},
		//{"SequentialGamma_linear_6" ,linear4},
		//{"SequentialGamma_linear_10" ,linear5},
		//{"SequentialGamma_linear_20" ,linear6},
		//{"RbfLinear_exp(journal)",  rbfLinear},
		//{"RbfLinear_exp",  rbfLinear},
		//{"RbfLinear_exp_coevolution(journal)",  rbfLinearCoevolution},
		//{"RbfLinear_exp_coevolution_min",  rbfLinearCoevolution_min},
		//{"RbfLinear_exp_coevolution_single_FS",  rbfLinearCoevolution_single},
		{"RbfLinear_exp_coevolution_max",  rbfLinearCoevolution_max},
		//{"RbfLinear_exp_coevolution_sum_2_kernels",  rbfLinearCoevolution_sum_2_kernels},
		{"RbfLinear_exp_coevolution_NoSMO_FS",  rbfLinearCoevolution_nosmo},
		{"RbfLinear_exp_coevolution_NoSMO",  rbfLinearCoevolution_nosmo},
		//{"ICPR_with_feature_selection", ICPR_WITH_FS},
		//{"AKSVM_with_feature_selection_max", MAX_WITH_FS},
		//{"SSVM_exp", genetic::DefaultSSVMConfig::getDefault() },
		/*{"RbfLinear_exp_coevolution_subset_50",  rbfLinearCoevolution50},
		{"RbfLinear_exp_coevolution_subset_20",  rbfLinearCoevolution20},
		{"RbfLinear_exp_coevolution_subset_10",  rbfLinearCoevolution10},*/
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs1)
	{

		configManager.setInitialNumberOfFeatures(config.second, info.numberOfFeatures);
		configManager.setupDataset(config.second, folder, config.first, outputFolder);
		configManager.setMetric(config.second);
		configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		//configManager.setSeedForRng(config.second, 12321);
		configManager.setupStopCondition(config.second);
		configManager.setNumberOfClasses(config.second, info.numberOfClasses);
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), config.first, fold);
	}
}

inline void CustomKernelExperiment(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	std::map<uint32_t, KernelParams> gridSearchResults; //not used in here
	auto outputResultsDir = std::filesystem::path(config.outputFolder); 


	std::vector<std::string> dataFolders = testApp::listDirectories(config.datafolder);
	std::map<std::string, testApp::DatasetInfo> datasetInfos;

	for (auto& folder : dataFolders)
	{
		datasetInfos[folder] = (testApp::getInfoAboutDataset(folder + "\\1\\train.csv"));
	}

	for (auto& folder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			//createConfigs1(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
			DifferentKernelConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
		}
	}

	//for 2D datasets only
	//svmComponents::DataNormalization::useDefinedMinMax(0, 500);

	//alter configs
	//changeConfigurationForExperiment(dataFolders, "SequentialGamma");


	dataFolders = testApp::listDirectories(outputResultsDir);
	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			//std::vector<std::string> filters{"SequentialGamma"};
			std::vector<std::string> filters{"MultipleGammaMASVM", "SequentialGamma" , "RbfLinear", "RbfLinearCoevolution", "SequentialGammaFS", "SSVM"};

			//consolidateExperiment(foldFolder, foldNumber, gridSearchResults, filters, true);
			runSpecified(foldFolder, foldNumber, gridSearchResults, filters, false, config.verbosity);
		}
	}

	//end
	return;
}



inline void manual_setting_experiment()
{
	auto dataFolder = R"(D:\PHD\experiments\A1_500\1)";

	auto config = genetic::CustomKernelConfig::getDefault();

	testApp::ConfigManager configManager;
//	configManager.setupSuperPixelsDataset(config, dataFolder, "CustomKernel");
	configManager.setRandomNumberGenerators(config);
	configManager.saveConfigToFileFolds(config, dataFolder, "CustomKernel", 1);
	

	config.putValue<std::string>("Svm.Metric", "AUC");
	svmComponents::DataNormalization::useDefinedMinMax(0, 500);

	changeConfigurationForExperiment({ R"(D:\PHD\experiments\A1_500\)" }, "CustomKernel");


	svmComponents::SvmTrainingCustomKernel training{ svmComponents::SvmAlgorithmConfiguration{config}, true, "RBF_CUSTOM", true };

	const auto con = genetic::SvmWokrflowConfiguration(config);
	genetic::LocalFileDatasetLoader loader(con.trainingDataPath, con.validationDataPath, con.testDataPath);

	testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath"));

	svmComponents::SvmCustomKernelChromosome manual;

	std::vector<svmComponents::Gene> chromosomeDataset;
	std::vector<int64_t> manualDataset{ 5,200,1000,510, 5020, 2765,3822 };
	std::vector<int64_t> gammas{500, 50, 100, 200, 50, 80, 80};
	std::vector<double> alphas{1000000, 1, 1, 1, -2, -500, -1};

	auto targets = loader.getTraningSet().getLabels();

	for(auto i = 0; i < manualDataset.size(); ++i)
	{
		chromosomeDataset.emplace_back(svmComponents::Gene(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]]), gammas[i]));
		std::cout << targets[manualDataset[i]] << std::endl;
	}

	manual = svmComponents::SvmCustomKernelChromosome(std::move(chromosomeDataset), 100);

	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop{ {manual} };

	training.trainPopulation(pop, loader.getTraningSet());

	auto svm2 = pop.getBestOne().getClassifier();

	auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm2.get());

	for(auto i = 0; i < res2->m_model->l; ++i)
	{
		std::cout << res2->m_model->sv_coef[0][i] << "\n";
	}

	*res2->m_model->sv_coef = alphas.data();

	svmComponents::CustomKernelEvolutionConfiguration algorithmConfig{ config, loader.getTraningSet() };
	svmStrategies::SvmValidationStrategy<svmComponents::SvmCustomKernelChromosome> validation{ *algorithmConfig.m_svmConfig.m_estimationMethod, false };

	pop = validation.launch(pop, loader.getTestSet());

	std::filesystem::path m_pngNameSource;
	genetic::setVisualizationFilenameAndFormat(algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, con, 0);

	auto svm = pop.getBestOne().getClassifier();

	auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());

	//*res->m_model->rho = -0.138962;

	svmComponents::SvmVisualization visualization;
	auto image = visualization.createDetailedVisualization(*res,
		algorithmConfig.m_svmConfig.m_height,
		algorithmConfig.m_svmConfig.m_width,
		loader.getTraningSet(), loader.getValidationSet());

	strategies::FileSinkStrategy savePngElement;

	savePngElement.launch(image, m_pngNameSource);

	std::cout << pop.getBestOne().getFitness();
}



inline void manual_setting_experiment2()
{
	auto dataFolder = R"(D:\PHD\experiments\2D_custom_gamma_check\1)";

	auto config = genetic::CustomKernelConfig::getDefault();

	testApp::ConfigManager configManager;
	//configManager.setupSuperPixelsDataset(config, dataFolder, "CustomKernel");
	configManager.setRandomNumberGenerators(config);
	configManager.saveConfigToFileFolds(config, dataFolder, "CustomKernel", 1);


	config.putValue<std::string>("Svm.Metric", "AUC");
	svmComponents::DataNormalization::useDefinedMinMax(0, 500);

	changeConfigurationForExperiment({ R"(D:\PHD\experiments\2D_custom_gamma_check\)" }, "CustomKernel");


	svmComponents::SvmTrainingCustomKernel training{ svmComponents::SvmAlgorithmConfiguration{config}, true, "RBF_CUSTOM", true };

	const auto con = genetic::SvmWokrflowConfiguration(config);
	genetic::LocalFileDatasetLoader loader(con.trainingDataPath, con.validationDataPath, con.testDataPath);

	testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath"));

	svmComponents::SvmCustomKernelChromosome manual;

	std::vector<svmComponents::Gene> chromosomeDataset;
	auto smallWidth = 1000;
	auto large = 10;
	std::vector<int64_t> manualDataset{ 1929, 1585, 3569, 615,3905, 3257,  3509, 1177, 3653, 593, 4209, 4437, 2093 ,3437, 1613, 2733, 957, 2461};
	std::vector<int64_t> gammas{ smallWidth, smallWidth, large, large, smallWidth, smallWidth, large, 
		smallWidth, smallWidth, large, smallWidth, smallWidth, smallWidth, smallWidth, smallWidth, large, smallWidth, smallWidth };

	//std::vector<int64_t> manualDataset{  1585, 3569,  3257,  3509, 1177,  /*593, 4209, 4437, 2733, 957,*/ };
	//std::vector<int64_t> gammas{smallWidth, large,smallWidth, large,smallWidth, /* large, smallWidth,    smallWidth, large, smallWidth,*/};

	/*std::vector<int64_t> manualDataset{ 1929, 1585,  3905, 3257,   1177, 3653,  4209, 4437, 2093 ,3437, 1613,  957, 2461 };
	std::vector<int64_t> gammas{ smallWidth, smallWidth, smallWidth, smallWidth,
		smallWidth, smallWidth, smallWidth, smallWidth, smallWidth, smallWidth, smallWidth,  smallWidth, smallWidth };*/

	//s�abe wektory
	std::vector<int64_t> manualDataset2{ 100, 928, 1088, 359, 391,1480,935,392,467,813,1468,1488 };
	std::vector<int64_t> gammas2{ smallWidth, smallWidth, smallWidth, smallWidth,
		smallWidth, smallWidth, smallWidth, smallWidth, smallWidth, smallWidth, smallWidth,  smallWidth };


	std::vector<std::uint64_t> best = { 524,3340,264,1224,4464,856,152,4496 };
	std::vector<int64_t> gammas_l{ large,large,large,large,large,large,large,large };
	
	//std::vector<int64_t> manualDataset{ 1929, 1585 /*3569*/, /*615*/3905, 3257,  /*3509*/ 1177, 3653,/*593*/ 4209, 4437, 2093 ,3437, 1613, /*2733*/ 957, 2461 };
	//std::vector<int64_t> gammas{ smallWidth, smallWidth, /*large*/ /*large*/ smallWidth, smallWidth, /*large*/
	//	smallWidth, smallWidth, /*large*/ smallWidth, smallWidth, smallWidth, smallWidth, smallWidth, /*large*/ smallWidth, smallWidth };


	//std::vector<double> alphas{ 1000000, 1, 1, 1, -2, -500, -1 };


	std::mt19937 eng1(1234);
	auto eng2 = eng1;

	std::shuffle(begin(manualDataset), end(manualDataset), eng1);
	std::shuffle(begin(gammas), end(gammas), eng2);

	auto targets = loader.getTraningSet().getLabels();
	
	for (auto i = 0; i < manualDataset2.size(); ++i)
	{
		//chromosomeDataset.emplace_back(svmComponents::Gene(manualDataset2[i], static_cast<std::uint8_t>(targets[manualDataset2[i]]), gammas[i]));
		//std::cout << targets[manualDataset[i]] << std::endl;
	}
	for (auto i = 0; i < manualDataset.size(); ++i)
	{
		chromosomeDataset.emplace_back(svmComponents::Gene(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]]), gammas[i]));
		//std::cout << targets[manualDataset[i]] << std::endl;
	}
	for (auto i = 0; i < best.size(); ++i)
	{
		//chromosomeDataset.emplace_back(svmComponents::Gene(best[i], static_cast<std::uint8_t>(targets[best[i]]), gammas_l[i]));
		//std::cout << targets[manualDataset[i]] << std::endl;
	}



	genetic::Timer timer;

	manual = svmComponents::SvmCustomKernelChromosome(std::move(chromosomeDataset), 1);

	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop{ {manual} };

	training.trainPopulation(pop, loader.getTraningSet());

	auto svm2 = pop.getBestOne().getClassifier();

	auto res2 = reinterpret_cast<phd::svm::libSvmImplementation*>(svm2.get());

	std::cout << "Alphas: \n";
	for (auto i = 0; i < res2->m_model->l; ++i)
	{
		std::cout << res2->m_model->sv_coef[0][i] << "\n";
	}

	//*res2->m_model->sv_coef = alphas.data();

	svmComponents::CustomKernelEvolutionConfiguration algorithmConfig{ config, loader.getTraningSet() };
	svmStrategies::SvmValidationStrategy<svmComponents::SvmCustomKernelChromosome> validation{ *algorithmConfig.m_svmConfig.m_estimationMethod, false };

	pop = validation.launch(pop, loader.getValidationSet());
	auto test_pop = pop;
	test_pop = validation.launch(test_pop, loader.getTestSet());

	std::filesystem::path m_pngNameSource;
	genetic::setVisualizationFilenameAndFormat(algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, con, 0);

	auto svm = pop.getBestOne().getClassifier();

	auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
	auto [sv_to_vec_map, scores] = res->check_sv(loader.getTraningSet());

	//*res->m_model->rho = -0.138962; //-0.138962;

	svmComponents::SvmVisualization visualization;
	visualization.setMap(sv_to_vec_map);
	visualization.setScores(scores);
	visualization.setGene(manual);
	auto image = visualization.createDetailedVisualization(*res,
		algorithmConfig.m_svmConfig.m_height,
		algorithmConfig.m_svmConfig.m_width,
		loader.getTraningSet(), loader.getValidationSet());

	strategies::FileSinkStrategy savePngElement;

	savePngElement.launch(image, m_pngNameSource);

	std::ofstream out(m_pngNameSource.string() + timeUtils::getTimestamp() + ".txt");
	out << "AUC: " << pop.getBestOne().getFitness() << std::endl;
	out << "Tr: ";
	for (auto v : manual.getDataset())
		out << v.id << " ";
	out << std::endl;

	out << "Gammas: ";
	for (auto v : manual.getDataset())
		out << v.gamma << " ";
	out << std::endl;

	out << "Alphas: ";
	for (auto i = 0; i < res2->m_model->l; ++i)
	{
		out << res2->m_model->sv_coef[0][i] << " ";
	}
	out << std::endl;

	out.close();
	std::cout << pop.getBestOne().getFitness();

	genetic::GeneticWorkflowResultLogger logger;
	auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
	auto validationDataset = loader.getValidationSet();
	auto featureNumber = validationDataset.getSamples()[0].size();

	logger.createLogEntry(pop, test_pop, timer, "ManualExperiment", 0,
		Accuracy(bestOneConfustionMatrix),
		featureNumber,
		bestOneConfustionMatrix);
	logger.logToFile(m_pngNameSource.string() + "log.txt");

}



inline void manual_setting_experiment_simple_rbf()
{
	auto dataFolder = R"(D:\PHD\experiments\2D_custom_gamma_check\1)";

	auto config = genetic::CustomKernelConfig::getDefault();

	testApp::ConfigManager configManager;
//	configManager.setupSuperPixelsDataset(config, dataFolder, "CustomKernel");
	//configManager.setRandomNumberGenerators(config);
	configManager.saveConfigToFileFolds(config, dataFolder, "CustomKernel", 1);

	config.putValue<std::string>("Svm.Metric", "AUC");
	//svmComponents::DataNormalization::useDefinedMinMax(0, 500);

	changeConfigurationForExperiment({ R"(D:\PHD\experiments\2D_custom_gamma_check\)" }, "CustomKernel");



	genetic::Timer timer;

	svmComponents::SvmKernelTraining training{ svmComponents::SvmAlgorithmConfiguration{config}, true };

	const auto con = genetic::SvmWokrflowConfiguration(config);
	genetic::LocalFileDatasetLoader loader(con.trainingDataPath, con.validationDataPath, con.testDataPath);

	testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath"));


	svmComponents::SvmKernelChromosome c(phd::svm::KernelTypes::Rbf, std::vector<double>{100.0, 1.0}, false);
	geneticComponents::Population<svmComponents::SvmKernelChromosome> pop(std::vector<svmComponents::SvmKernelChromosome> {c});


	training.trainPopulation(pop, loader.getTraningSet());

	svmComponents::CustomKernelEvolutionConfiguration algorithmConfig{ config, loader.getTraningSet() };
	svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome> validation{ *algorithmConfig.m_svmConfig.m_estimationMethod, false };

	pop = validation.launch(pop, loader.getTestSet());

	std::filesystem::path m_pngNameSource;
	genetic::setVisualizationFilenameAndFormat(algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, con, 0);

	auto svm = pop.getBestOne().getClassifier();

	svmComponents::SvmVisualization visualization;
	auto image = visualization.createDetailedVisualization(*svm,
		algorithmConfig.m_svmConfig.m_height,
		algorithmConfig.m_svmConfig.m_width,
		loader.getTraningSet(), loader.getValidationSet());

	strategies::FileSinkStrategy savePngElement;

	savePngElement.launch(image, m_pngNameSource);

	std::cout << pop.getBestOne().getFitness();

	genetic::GeneticWorkflowResultLogger logger;
	auto bestOneConfustionMatrix = pop.getBestOne().getConfusionMatrix().value();
	auto validationDataset = loader.getValidationSet();
	auto featureNumber = validationDataset.getSamples()[0].size();

	logger.createLogEntry(pop, pop, timer, "ManualExperiment", 0,
		Accuracy(bestOneConfustionMatrix),
		featureNumber,
		bestOneConfustionMatrix);
	logger.logToFile(m_pngNameSource.string() + "log.txt");

}