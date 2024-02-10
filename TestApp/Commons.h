#pragma once

#include <algorithm>
#include <string>
#include <exception>
#include "ConfigParser.h"
#include "AppUtils/AppUtils.h"
#include "ConfigGeneration.h"


#include "libGeneticSvm/DefaultWorkflowConfigs.h"
#include "AppUtils/PythonFeatureSelection.h"
#include "libPlatform/loguru.hpp"
#include "libPlatform/TimeUtils.h"

enum class AlgorithmName
{
	GridSearch,
	GridSearchWithFetureSelection,
	GASVM,
	MASVM,
	//RandomSearch,
	ALGA,
	ALMA,
	FSALMA,
	ESVM,
	SE_SVM,
	SE_SVM_CORRECTED, //changed parent selection scheme to have same parents for all parts of chromosome, seems to not impact end results
	AKSVM, //ICPR 2021
	DA_SVM_MIN,
	DA_SVM_AVG,
	DA_SVM_MAX,
	DA_SVM_SUM,
	DA_SVM_NO_T,
	DA_SVM_CE_ONE,
	DA_SVM_CE_AVG,
	DA_SVM_CE_MAX,
	DA_SVM_CE_MIN,
	DA_SVM_CE_SUM,
	DA_SVM_CE_NO_T,
	DA_SVM_CE_FS_ONE,
	DA_SVM_CE_FS_MAX,
	DA_SVM_CE_FS_NO_T,

};

struct KernelParams
{
	KernelParams()
		: C(-1)
		, Gamma(-1)
	{
	}

	KernelParams(double c, double gamma)
		: C(c)
		, Gamma(gamma)
	{
	}

	double C;
	double Gamma;
};


void runSpecified(std::filesystem::path pathToFold,
                  uint32_t foldNumber,
                  std::map<uint32_t, KernelParams>& gridSearchResults,
                  std::vector<std::string>& filters,
                  bool wihtFeatureSelection = true,
                  testApp::Verbosity verbosity = testApp::Verbosity::All);

void consolidateExperimentCLASH(std::filesystem::path pathToFold,
                                uint32_t foldNumber,
                                std::map<uint32_t, KernelParams>& gridSearchResults,
                                std::vector<std::string>& filters,
                                bool withFeatureSelection);

inline void saveGridSearchResultsToFile(std::map<uint32_t, KernelParams>& gridSearchResults, std::filesystem::path datasetFolder, std::string algorithmName)
{
	std::ofstream tfStream(datasetFolder.string() + "\\" + algorithmName + ".txt");
	//std::ofstream tfStream(datasetFolder.string() + "\\" + timeUtils::getTimestamp() + "__" + algorithmName + ".txt");

	tfStream << "# Fitness \t C \t Gamma\n";
	
	for (const auto& value : gridSearchResults)
	{
		tfStream << value.first << " " << value.second.C << " " << value.second.Gamma << std::endl;
	}
	tfStream.close();

	gridSearchResults.clear();
}

inline void loadGridSearchParams(std::map<uint32_t, KernelParams>& gridSearchResults, std::filesystem::path datasetFolder, std::string algorithmName)
{
	gridSearchResults.clear();
	std::ifstream tfStream(datasetFolder.string() + "\\" + algorithmName + ".txt");

	if (tfStream.is_open())
	{
		uint32_t foldNumber;
		double C;
		double Gamma;

	std::string ommitHeader;
	std::getline(tfStream, ommitHeader);
	
		while (tfStream >> foldNumber >> C >> Gamma)
		{

			gridSearchResults[foldNumber] = KernelParams(C, Gamma);
		}
		tfStream.close();
	}
}

inline bool isNumber(const std::string& s)
{
	return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
}

inline uint32_t getFoldNumberFromFolder(const std::filesystem::path folder)
{

	auto value = folder.stem();

	if (isNumber(value.string()))
		return std::stoi(value);
	else
		throw std::exception(std::string("Fold folder consist non digit characters. Path: " + folder.string()).c_str());
}






inline void processConfigurationsAndSave(std::vector<std::pair<std::string, platform::Subtree>>& configs,
                                         std::string folder, uint32_t fold,
                                         std::string outputFolder,
                                         KernelParams parameters,
                                         testApp::DatasetInfo info,
                                         bool featureSelection = false)
{
	testApp::ConfigManager configManager;
	for (auto& config : configs)
	{
		if(featureSelection && info.numberOfFeatures > 4)
		{
			configManager.setInitialNumberOfFeatures(config.second, 4);
		}
		else
		{
			configManager.setInitialNumberOfFeatures(config.second, info.numberOfFeatures);
		}
		configManager.setupDataset(config.second, folder, config.first, outputFolder);
		configManager.setMetric(config.second);
		configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		configManager.setupStopCondition(config.second);
		configManager.addKernelParameters(config.second, parameters.C, parameters.Gamma);
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), config.first, fold);
	}
}

inline void createFSAlmaConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	//need to be used with runSpecified where featureSelection is set to true
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"FSALMA", genetic::DefaultAlgaConfig::getALMA()},
	};
	configs1[0].second.putValue("Name", "FSAlma"); //fix for running Alma twice when FSAlma run
	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info); //kernel params are default from memetic and evolved in algorithm
}

inline void createAlmaConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"ALMA", genetic::DefaultAlgaConfig::getALMA()},
	};
	
	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info); //kernel params are default from memetic and evolved in algorithm
}

inline void createESVMConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{		
		{"ESVM", genetic::DefaultKTFConfig::getDefault() }  //otherwise named as KTF
	};
	
	auto featureSelection = true;
	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info, featureSelection); //kernel params are default from memetic and evolved in algorithm
}

inline void createSESVMConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"SE-SVM", genetic::DefaultSSVMConfig::getDefault() }
	};
	
	auto featureSelection = true;
	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info, featureSelection); //kernel params are default from memetic and evolved in algorithm
}


inline void createSESVMCorrectedConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"SE-SVM_Corrected", genetic::DefaultSESVMCorrectedConfig::getDefault() }
	};

	auto featureSelection = true;
	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info, featureSelection); //kernel params are default from memetic and evolved in algorithm
}


inline void createAKSVMConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"AKSVM", genetic::DefaultSequentialGammaConfig::getDefault() }
	};

	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info); //kernel params are default from memetic and evolved in algorithm
}

inline void createDASVMConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	auto max = genetic::DefaultSequentialGammaConfig::getDefault();
	max.putValue<std::string>("Svm.SequentialGamma.KernelType", "RBF_MAX");
	
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"DASVM_MAX", max }
	};

	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info); //kernel params are default from memetic and evolved in algorithm
}



inline void createDASVM_Variants_Configs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info, AlgorithmName algorithmToRun)
{
	auto rbfLinearCoevolution_max = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_max.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_max.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_max.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_max.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR_MAX");

	auto rbfLinearCoevolution_single = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_single.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_single.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_single.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_single.putValue<std::string>("Svm.RbfLinear.KernelType", "RBF_LINEAR_SINGLE");

	auto rbfLinearCoevolution_nosmo = genetic::DefaultRbfLinearConfig::getDefault();
	rbfLinearCoevolution_nosmo.putValue<unsigned int>("Svm.RbfLinear.NumberOfClassExamples", info.numberOfFeatures);
	rbfLinearCoevolution_nosmo.putValue<std::string>("Svm.RbfLinear.SelectionOperator.Name", "ConstatntTruncationSelection");
	rbfLinearCoevolution_nosmo.putValue("Name", "RbfLinearCoevolution");
	rbfLinearCoevolution_nosmo.putValue<bool>("Svm.RbfLinear.TrainAlpha", false);



	std::vector<std::pair<std::string, platform::Subtree>> configs1 = {};


	if (algorithmToRun == AlgorithmName::DA_SVM_CE_FS_MAX)
	{
		rbfLinearCoevolution_max.putValue("Name", "RbfLinearCoevolutionFSCEMAX");
		auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_FS_MAX", rbfLinearCoevolution_max };
		configs1.emplace_back(p);
	}
	else if (algorithmToRun == AlgorithmName::DA_SVM_CE_FS_NO_T)
	{
		rbfLinearCoevolution_nosmo.putValue("Name", "RbfLinearCoevolutionFSCENOT");
		auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_FS_NO_T",  rbfLinearCoevolution_nosmo };
		configs1.emplace_back(p);
	}
	else if (algorithmToRun == AlgorithmName::DA_SVM_CE_FS_ONE)
	{
		rbfLinearCoevolution_single.putValue("Name", "RbfLinearCoevolutionFSCEONE");
		auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_FS_ONE",  rbfLinearCoevolution_single };
		configs1.emplace_back(p);
	}
	else if (algorithmToRun == AlgorithmName::DA_SVM_CE_MAX)
	{
		rbfLinearCoevolution_max.putValue("Name", "RbfLinearCoevolutionCEMAX");
		auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_MAX", rbfLinearCoevolution_max };
		configs1.emplace_back(p);
	}
	else if (algorithmToRun == AlgorithmName::DA_SVM_CE_NO_T)
	{
		rbfLinearCoevolution_nosmo.putValue("Name", "RbfLinearCoevolutionCENOT");
		auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_NO_T",  rbfLinearCoevolution_nosmo };
		configs1.emplace_back(p);
	}
	else if (algorithmToRun == AlgorithmName::DA_SVM_CE_ONE)
	{
		rbfLinearCoevolution_single.putValue("Name", "RbfLinearCoevolutionCEONE");
		auto p = std::pair<std::string, platform::Subtree>{ "DA_SVM_CE_ONE",  rbfLinearCoevolution_single };
		configs1.emplace_back(p);
	}
	else
	{
		throw std::exception("Selected variant of DASVM algorithm is not supported right now");
	}

	processConfigurationsAndSave(configs1, folder, fold, outputFolder, KernelParams(1, 1), info); //kernel params are default from memetic and evolved in algorithm
}


inline void createAlgaConfigs(std::string folder, uint32_t fold, const std::vector<uint32_t>& Kvalues, std::string outputFolder)
{
	std::vector<std::pair<std::string, platform::Subtree>> configsAlga =
	{
		{"ALGA", genetic::DefaultAlgaConfig::getDefault()},
	};
	testApp::ConfigManager configManager;
	for (auto& config : configsAlga)
	{
		for (auto K : Kvalues)
		{
			const auto algorithmName = config.first + "_" + std::to_string(K) + "_";
			
			configManager.setupDataset(config.second, folder, algorithmName, outputFolder);
			configManager.setMetric(config.second);
			configManager.setGridKernelInitialPopulationGeneration(config.second);
			configManager.setRandomNumberGenerators(config.second);
			configManager.setK(config.second, K);
			configManager.setupStopCondition(config.second);
			std::vector<std::string> elements = testApp::splitPath(folder);
			configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), algorithmName, fold);
		}
	}
}

inline void createGasvmConfigs(std::string folder, uint32_t fold, const std::vector<uint32_t>& Kvalues, std::string outputFolder, KernelParams parameters)
{
	std::vector<std::pair<std::string, platform::Subtree>> configsAlga =
	{
		{"GASVM", genetic::DefaultGaSvmConfig::getDefault()},
	};
	testApp::ConfigManager configManager;
	for (auto& config : configsAlga)
	{
		for (auto K : Kvalues)
		{
			const auto algorithmName = config.first + "_" + std::to_string(K) + "_";

			configManager.setupDataset(config.second, folder, algorithmName, outputFolder);
			configManager.setMetric(config.second);
			configManager.setGridKernelInitialPopulationGeneration(config.second);
			configManager.setRandomNumberGenerators(config.second);
			configManager.setK(config.second, K);
			configManager.addKernelParameters(config.second, parameters.C, parameters.Gamma);
			configManager.setupStopCondition(config.second);
			std::vector<std::string> elements = testApp::splitPath(folder);
			configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), algorithmName, fold);
		}
	}
}

inline void createMasvmConfiguration(std::string folder, uint32_t fold, std::string outputFolder, KernelParams parameters)
{
	auto MASVM_K4 = genetic::DefaultMemeticConfig::getDefault();
	MASVM_K4.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 4);
	auto MASVM_K16 = genetic::DefaultMemeticConfig::getDefault();
	MASVM_K16.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 16);
	auto MASVM_K32 = genetic::DefaultMemeticConfig::getDefault();
	MASVM_K32.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 32);
	auto MASVM_K64 = genetic::DefaultMemeticConfig::getDefault();
	MASVM_K64.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 64);

	std::vector<std::pair<std::string, platform::Subtree>> configs2 =
	{
		{"MASVM", genetic::DefaultMemeticConfig::getDefault()},
		/*{"MASVM_K=4", MASVM_K4},
		{"MASVM_K=16", MASVM_K16},
		{"MASVM_K=32", MASVM_K32},
		{"MASVM_K=64", MASVM_K64},*/
		//{"FeaturesSet", genetic::DefaultFeatureSelectionConfig::getDefault()},
		//{"TF", genetic::DefaultTFConfig::getDefault()},
		//{"FT", genetic::DefaultFTConfig::getDefault()},
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs2)
	{
		configManager.setupDataset(config.second, folder, config.first, outputFolder);
		configManager.setMetric(config.second);
		configManager.setRandomNumberGenerators(config.second);
		//configManager.setK(config.second, numberOfFeatures);
		configManager.addKernelParameters(config.second, parameters.C, parameters.Gamma);
		configManager.setupStopCondition(config.second);
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), config.first, fold);
	}
}







inline void createEnsembleConfigs(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	auto linear = genetic::DefaultEnsembleTreeConfig::getDefault();
	linear.putValue<std::string>("Svm.KernelType", "LINEAR");
	linear.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", info.numberOfFeatures);
	linear.putValue<bool>("Svm.EnsembleTree.ConstKernel", true);

	auto rbf_maxK = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_maxK.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 8);

	

	/*zeroOut,*/
	
	
	auto linearMetric = genetic::DefaultEnsembleTreeConfig::getDefault();
	linearMetric.putValue<std::string>("Svm.KernelType", "LINEAR");
	linearMetric.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", info.numberOfFeatures);
	linearMetric.putValue<bool>("Svm.EnsembleTree.ConstKernel", true);
	linearMetric.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	
	auto rbf_maxK_metric = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_maxK_metric.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 8);
	rbf_maxK_metric.putValue<bool>("Svm.EnsembleTree.SwitchFitness", true);
	rbf_maxK_metric.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	
	auto rbf_metric = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_metric.putValue<bool>("Svm.EnsembleTree.SwitchFitness", true);
	rbf_metric.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");



	//nonlinearDecrease,
	

	auto linearMetric1 = genetic::DefaultEnsembleTreeConfig::getDefault();
	linearMetric1.putValue<std::string>("Svm.KernelType", "LINEAR");
	linearMetric1.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", info.numberOfFeatures);
	linearMetric1.putValue<bool>("Svm.EnsembleTree.ConstKernel", true);
	linearMetric1.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "nonlinearDecrease");

	auto rbf_maxK_metric1 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_maxK_metric1.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 8);
	rbf_maxK_metric1.putValue<bool>("Svm.EnsembleTree.SwitchFitness", true);
	rbf_maxK_metric1.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "nonlinearDecrease");

	auto rbf_metric1 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_metric1.putValue<bool>("Svm.EnsembleTree.SwitchFitness", true);
	rbf_metric1.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "nonlinearDecrease");

	//	boundryCheck
	auto linearMetric2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	linearMetric2.putValue<std::string>("Svm.KernelType", "LINEAR");
	linearMetric2.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", info.numberOfFeatures);
	linearMetric2.putValue<bool>("Svm.EnsembleTree.ConstKernel", true);
	linearMetric2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "boundryCheck");

	auto rbf_maxK_metric2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_maxK_metric2.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", 8);
	rbf_maxK_metric2.putValue<bool>("Svm.EnsembleTree.SwitchFitness", true);
	rbf_maxK_metric2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "boundryCheck");

	auto rbf_metric2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_metric2.putValue<bool>("Svm.EnsembleTree.SwitchFitness", true);
	rbf_metric2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "boundryCheck");

	auto rbfM = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfM.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true); 
	rbfM.putValue<std::string>("Svm.EnsembleTree.SvMode", "previous");

	auto rbfG = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfG.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");

	auto linearG = genetic::DefaultEnsembleTreeConfig::getDefault();
	linearG.putValue<std::string>("Svm.KernelType", "LINEAR");
	linearG.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", info.numberOfFeatures);
	linearG.putValue<bool>("Svm.EnsembleTree.ConstKernel", true);
	linearG.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	linearG.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");


	auto linearLP = genetic::DefaultEnsembleTreeConfig::getDefault();
	linearLP.putValue<std::string>("Svm.KernelType", "LINEAR");
	linearLP.putValue<double>("Svm.MemeticTrainingSetSelection.Memetic.MaxK", info.numberOfFeatures);
	linearLP.putValue<bool>("Svm.EnsembleTree.ConstKernel", true);
	linearLP.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	linearLP.putValue<std::string>("Svm.EnsembleTree.SvMode", "previous");


	auto rbfAddToTraining = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfAddToTraining.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfAddToTraining.putValue<std::string>("Svm.EnsembleTree.SvMode", "previous");

	auto rbfAddToTrainingAll = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfAddToTrainingAll.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfAddToTrainingAll.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");


	auto rbfAddToTrainingAll2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfAddToTrainingAll2.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfAddToTrainingAll2.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	rbfAddToTrainingAll2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");


	auto rbfAddToTraining2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfAddToTraining2.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfAddToTraining2.putValue<std::string>("Svm.EnsembleTree.SvMode", "previous");
	rbfAddToTraining2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");

	auto rbfM2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfM2.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfM2.putValue<std::string>("Svm.EnsembleTree.SvMode", "previous");
	rbfM2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");

	auto rbfG2 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfG2.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	rbfG2.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
	rbfG2.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);


	auto rbf_light = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_light.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbf_light.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbf_light.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbf_light.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbf_light.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	

	auto rbfG4 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG4.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfG4.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	rbfG4.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG4.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
	rbfG4.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", false);

	auto rbfG3 = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG3.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfG3.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	rbfG3.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");

	auto rbfG2_ir = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_ir.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfG2_ir.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	rbfG2_ir.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_ir.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
	rbfG2_ir.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_ir.putValue<bool>("Svm.EnsembleTree.UseImbalanceRatio", true);
	rbfG2_ir.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);

	auto rbf_light_ir = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_light_ir.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbf_light_ir.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbf_light_ir.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbf_light_ir.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbf_light_ir.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbf_light_ir.putValue<bool>("Svm.EnsembleTree.UseImbalanceRatio", true);

	auto rbf_fs = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbf_fs.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbf_fs.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbf_fs.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbf_fs.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbf_fs.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbf_fs.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);

	auto rbfG2_fs = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_fs.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	rbfG2_fs.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	rbfG2_fs.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_fs.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
	rbfG2_fs.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_fs.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);


	auto rbfG2_newFlow = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_newFlow.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfG2_newFlow.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbfG2_newFlow.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_newFlow.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbfG2_newFlow.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_newFlow.putValue<bool>("Svm.EnsembleTree.NewDatasetSampling", true);

	auto rbfG2_newFlowFS = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_newFlowFS.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfG2_newFlowFS.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbfG2_newFlowFS.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_newFlowFS.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbfG2_newFlowFS.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_newFlowFS.putValue<bool>("Svm.EnsembleTree.NewDatasetSampling", true);
	rbfG2_newFlowFS.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);


	auto rbfG2_newFlowTR = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_newFlowTR.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfG2_newFlowTR.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbfG2_newFlowTR.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_newFlowTR.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbfG2_newFlowTR.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_newFlowTR.putValue<bool>("Svm.EnsembleTree.NewDatasetSampling", true);
	rbfG2_newFlowTR.putValue<bool>("Svm.EnsembleTree.NewSamplesForTraining", true);


	auto rbfG2_newFlowChangeResampling = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_newFlowChangeResampling.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfG2_newFlowChangeResampling.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbfG2_newFlowChangeResampling.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_newFlowChangeResampling.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbfG2_newFlowChangeResampling.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_newFlowChangeResampling.putValue<bool>("Svm.EnsembleTree.NewDatasetSampling", true);
	rbfG2_newFlowChangeResampling.putValue<bool>("Svm.EnsembleTree.NewSamplesForTraining", false);
	rbfG2_newFlowChangeResampling.putValue<bool>("Svm.EnsembleTree.ResamplingWithNoAddition", true);

	auto rbfG2_newFlowChangeResamplingFS = genetic::DefaultEnsembleTreeConfig::getDefault();
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", false);
	rbfG2_newFlowChangeResamplingFS.putValue<std::string>("Svm.EnsembleTree.SvMode", "defaultOption");
	rbfG2_newFlowChangeResamplingFS.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.DasvmKernel", false);
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.NewDatasetSampling", true);
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.NewSamplesForTraining", false);
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.ResamplingWithNoAddition", true);
	rbfG2_newFlowChangeResamplingFS.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);


	//config.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", false);
	//config.putValue<bool>("Svm.EnsembleTree.UseBias", false);
	//config.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", false);
	
	auto newFlow_with_class = rbfG2_newFlowChangeResampling;
	newFlow_with_class.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);


	auto newFlow_with_class_bias = rbfG2_newFlowChangeResampling;
	newFlow_with_class_bias.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	newFlow_with_class_bias.putValue<bool>("Svm.EnsembleTree.UseBias", true);

	auto newFlow_with_class_bias_V = rbfG2_newFlowChangeResampling;
	newFlow_with_class_bias_V.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	newFlow_with_class_bias_V.putValue<bool>("Svm.EnsembleTree.UseBias", true);
	newFlow_with_class_bias_V.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", true);


	auto newFlow_with_class_bias_V_DASVM = rbfG2_newFlowChangeResampling;
	newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.UseBias", true);
	newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", false);
	newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.AddSvToTraining", true);
	newFlow_with_class_bias_V_DASVM.putValue<std::string>("Svm.EnsembleTree.SvMode", "global");
	newFlow_with_class_bias_V_DASVM.putValue<std::string>("Svm.EnsembleTree.GrowKMode", "zeroOut");
	newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.DasvmKernel", true);
	newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.AddAlmaNode", true);
	//newFlow_with_class_bias_V_DASVM.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true); //last change

	
	auto newFlow_V = rbfG2_newFlowChangeResampling;
	newFlow_V.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", true);


	auto newFlow_class_V = rbfG2_newFlowChangeResampling;
	newFlow_class_V.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	newFlow_class_V.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", true);

	auto newFlow_with_bias_V = rbfG2_newFlowChangeResampling;
	newFlow_with_bias_V.putValue<bool>("Svm.EnsembleTree.UseBias", true);
	newFlow_with_bias_V.putValue<bool>("Svm.EnsembleTree.NewFlowFullValidationSet", true);



	auto newFlow_with_class_bias_V_mod = newFlow_with_class_bias_V;
	newFlow_with_class_bias_V_mod.putValue<bool>("Svm.EnsembleTree.NewSamplesForTraining", true);
	newFlow_with_class_bias_V_mod.putValue<bool>("Svm.EnsembleTree.ResamplingWithNoAddition", false);

	//rbfAddToTrainingAll2.putValue<bool>("Svm.EnsembleTree.UseBias", true);
	//rbfG2.putValue<bool>("Svm.EnsembleTree.UseBias", true);
	//rbf_light.putValue<bool>("Svm.EnsembleTree.UseBias", true);
	//rbfG3.putValue<bool>("Svm.EnsembleTree.UseBias", true);

	//rbfAddToTrainingAll2.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	//rbfG2.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	//rbf_light.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);
	//rbfG3.putValue<bool>("Svm.EnsembleTree.UseSingleClassThresholds", true);

	auto bigSetsEnsemble = genetic::DefaultBigSetsEnsembleConfig::getDefault();
	
	auto bigSetsEnsembleFS = genetic::DefaultBigSetsEnsembleConfig::getDefault();
	bigSetsEnsembleFS.putValue<bool>("Svm.EnsembleTree.UseFeatureSelction", true);

	auto bigSetsEnsembleCascadeFS = genetic::DefaultBigSetsEnsembleConfig::getDefault();
	bigSetsEnsembleCascadeFS.putValue<bool>("Svm.EnsembleTree.UseFeatureSelctionCascadeWise", true);
	
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"BigSetsEnsemble_v8", bigSetsEnsemble},
		//{"BigSetsEnsemble_v8_NO_LINEAR", bigSetsEnsemble}, // CHANGE IN BIG SETS BigSetsEnsemble.cpp line 585 LINEAR SVM TRAINING and 617
		//{"BigSetsEnsemble_v8_FS", bigSetsEnsembleFS},

		//{"BigSetsEnsemble_v8_Cascade_FS", bigSetsEnsembleCascadeFS}, //TODO implement
		
		//{"EnsembleALMA", genetic::DefaultEnsembleConfig::getDefault()},
		//{"Baseline_EnsembleList_RBF", genetic::DefaultEnsembleTreeConfig::getDefault()},
		//{"EnsembleList_Linear", linear},
		//{"EnsembleList_RBF_MaxK=8", rbf_maxK},
		//{"EnsembleList_RBF_ZeroOut", rbf_metric},
		//{"EnsembleList_Linear_ZeroOut", linearMetric},


		//{"AllSV_EnsembleList_RBF_ZeroOut", rbfAddToTrainingAll2},
		//{"PreviousSV_EnsembleList_RBF_ZeroOut", rbfAddToTraining2},
		
			//{"InsertAllSV_EnsembleList_RBF_ZeroOut_DASVM", rbfG2},
		//{"EnsembleList_With_SESVM", rbfG2}, //GECCO 2022 !!!!!!!!!!!
		//{"EnsembleList_With_Alma_NewFlow", rbfG2_newFlow},
		//{"EnsembleList_With_Alma_NewFlow_FS", rbfG2_newFlowFS},
		//{"EnsembleList_With_Alma_NewFlow_TrainingAdd", rbfG2_newFlowTR},
		//{"EnsembleList_With_Alma_NewFlow_NoAddSample", rbfG2_newFlowChangeResampling},


		//{"EnsembleList_ET_NF_Baseline", rbfG2_newFlowChangeResampling},
		//{"EnsembleList_ET_NF_Class", newFlow_with_class},
		//{"EnsembleList_ET_NF_Class_Bias", newFlow_with_class_bias},
		//{"EnsembleList_ET_NF_Class_Bias_V", newFlow_with_class_bias_V},
		//{"EnsembleList_ET_NF_Class_Bias_V_mod", newFlow_with_class_bias_V_mod},
		//{"EnsembleList_ET_NF_Class_Bias_V_Inheritance", newFlow_with_class_bias_V_DASVM},
		
		//{"EnsembleList_ET_NF_Baseline_V", newFlow_V},
		//{"EnsembleList_ET_NF_Class_V", newFlow_class_V},
		//{"EnsembleList_ET_NF_Baseline_Bias_V", newFlow_with_bias_V},

		
		//{"EnsembleList_With_Alma_NewFlow_NoAddSampleFS", rbfG2_newFlowChangeResamplingFS},
		
		//{"EnsembleList_DistanceScheme", rbfG4},
			//{"EnsembleList_ALMA_no_inheritance", rbf_light},
		//{"EnsembleList_ALMA_no_inheritance_FS", rbf_fs},

		//{"EnsembleList_With_Alma_FS", rbfG2_fs},
		//{"EnsembleList_With_Alma_IR_FS", rbfG2_ir},
		//{"EnsembleList_ALMA_no_inheritance_IR", rbf_light_ir},
		
		
			//{"InsertAllSV_EnsembleList_RBF_ZeroOut", rbfG3},
		//{"InsertPreviousSV_EnsembleList_RBF_ZeroOut", rbfM2},

		//{"AllSV_EnsembleList_RBF", rbfAddToTrainingAll},
		//{"PreviousSV_EnsembleList_RBF", rbfAddToTraining},
		//
		//{"InsertAllSV_EnsembleList_RBF", rbfG},
		//{"InsertPreviousSV_EnsembleList_RBF", rbfM},

		//{"InsertAllSV_EnsembleList_Linear", linearG},
		//{"InsertPreviousSV_EnsembleList_Linear", linearLP},

		
		
		//{"EnsembleList_RBF_nonlinearDecrease", rbf_metric1},
		//{"EnsembleList_Linear_nonlinearDecrease", linearMetric1},
		//
		//{"EnsembleList_RBF_boundryCheck", rbf_metric2},
		//{"EnsembleList_Linear_boundryCheck", linearMetric2},



		//{"EnsembleList_RBF_MaxK=8_nonlinearDecrease", rbf_maxK_metric1},
		//{"EnsembleList_RBF_MaxK=8_ZeroOut", rbf_maxK_metric},
		////{"EnsembleList_RBF_MaxK=8_boundryCheck", rbf_maxK_metric2},
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs1)
	{
		if (info.numberOfFeatures > 4)
		{
			configManager.setInitialNumberOfFeatures(config.second, 4);
		}
		else
		{
			configManager.setInitialNumberOfFeatures(config.second, info.numberOfFeatures);
		}

		configManager.setupDataset(config.second, folder, config.first, outputFolder);
		configManager.setMetric(config.second);
		configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		configManager.setupStopCondition(config.second);
		configManager.setNumberOfClasses(config.second, info.numberOfClasses);
		if(config.first == "EnsembleList_RBF_MaxK=8")
		{
			configManager.setK(config.second, 2);  //start from the number of features in dataset -- probably need to be checked for some datasets (with many features)
		}
		else
		{
			//configManager.setK(config.second, info.numberOfFeatures);  //start from the number of features in dataset -- probably need to be checked for some datasets (with many features)
			configManager.setK(config.second, 8);  //start from const number
		}
		
		
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), config.first, fold);
	}
}

inline void createConfigs1(std::string folder, uint32_t fold, std::string outputFolder, testApp::DatasetInfo info)
{
	auto ALMA_K4 = genetic::DefaultAlgaConfig::getALMA();
	ALMA_K4.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 4);
	auto ALMA_K16 = genetic::DefaultAlgaConfig::getALMA();
	ALMA_K16.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 16);
	auto ALMA_K32 = genetic::DefaultAlgaConfig::getALMA();
	ALMA_K32.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 32);
	auto ALMA_K64 = genetic::DefaultAlgaConfig::getALMA();
	ALMA_K64.putValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples", 64);
	
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		//{"ALMA_K=8", genetic::DefaultAlgaConfig::getALMA()},
		/*{"ALMA_K=4", ALMA_K4},
		{"ALMA_K=16", ALMA_K16},
		{"ALMA_K=32", ALMA_K32},
		{"ALMA_K=64", ALMA_K64},*/
		///*{ "Kernel", genetic::DefaultKernelEvolutionConfig::getDefault() },*/
		//{"KTF", genetic::DefaultKTFConfig::getDefault()},
		{"SSVM", genetic::DefaultSSVMConfig::getDefault() },
		/*{ "RandomSearch", genetic::DefaultRandomSearchConfig::getDefault() },
		{ "RandomSearchInitPop", genetic::DefaultRandomSearchInitPopConfig::getDefault() },
		{ "RandomSearchEvoHelp", genetic::DefaultRandomSearchEvoHelpConfig::getDefault() },*/
		//{"CustomKernel", genetic::CustomKernelConfig::getDefault()},
		//{"SequentialGamma", genetic::DefaultSequentialGammaConfig::getDefault()},
		//{"MultipleGammaMASVM", genetic::DefaultMultipleGammaMASVMConfig::getDefault()},
		//{"EnsembleALMA", genetic::DefaultEnsembleConfig::getDefault()},
		//{"EnsembleList", genetic::DefaultEnsembleTreeConfig::getDefault()},
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs1)
	{
		configManager.setInitialNumberOfFeatures(config.second, info.numberOfFeatures);
		configManager.setupDataset(config.second, folder, config.first, outputFolder);
		configManager.setMetric(config.second);
		configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		configManager.setupStopCondition(config.second);
		configManager.setNumberOfClasses(config.second, info.numberOfClasses);
		//configManager.setK(config.second, 8);  //8 is the default setting
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second,  outputFolder + "\\" + *(elements.end()-2) + "\\" + *(elements.end()-1), config.first, fold);
	}
}

inline void gridSearchWithOutFeatures(std::string folder, uint32_t fold, std::string outputFolder)
{
	auto poly = genetic::DefaultGridSearchConfig::getDefault();
	poly.putValue("Svm.KernelType", "POLY");

	auto linear = genetic::DefaultGridSearchConfig::getDefault();
	linear.putValue("Svm.KernelType", "LINEAR");


	auto full_rbf_overfitted = genetic::DefaultGridSearchConfig::getDefault();
	full_rbf_overfitted.putValue<double>("GridSearch.gammaGrid.Min", 500000.0000);
	full_rbf_overfitted.putValue<double>("GridSearch.gammaGrid.Max", 500001.0001);

	full_rbf_overfitted.putValue<double>("GridSearch.cGrid.Min", 1000.0001); 
	full_rbf_overfitted.putValue<double>("GridSearch.cGrid.Max", 1000.1);  


	auto rbf_poly = genetic::DefaultGridSearchConfig::getDefault();
	rbf_poly.putValue("Svm.KernelType", "RBF_POLY_GLOBAL");

	rbf_poly.putValue<double>("GridSearch.gammaGrid.Min", 0.0001);
	rbf_poly.putValue<double>("GridSearch.gammaGrid.Max", 1000.1);

	rbf_poly.putValue<double>("GridSearch.cGrid.Min", 0.0001);
	rbf_poly.putValue<double>("GridSearch.cGrid.Max", 1000.1);

	
	rbf_poly.putValue<double>("GridSearch.degreeGrid.Min", 2);
	rbf_poly.putValue<double>("GridSearch.degreeGrid.Max", 7);
	rbf_poly.putValue<double>("GridSearch.degreeGrid.LogStep", 10); //this value is not used at all!!!

	rbf_poly.putValue<double>("GridSearch.tGrid.Min", 0.1); 
	rbf_poly.putValue<double>("GridSearch.tGrid.Max", 0.9);  
	rbf_poly.putValue<double>("GridSearch.tGrid.LogStep", 1);  //this value is not used at all!!!


	auto gridSearchConfig = genetic::DefaultGridSearchConfig::getDefault();
	gridSearchConfig.putValue<int>("GridSearch.NumberOfIteratrions", 1);
	gridSearchConfig.putValue<int>("GridSearch.NumberOfFolds", 1);
	gridSearchConfig.putValue<double>("GridSearch.cGrid.Min", 0.001); //0.0000000000000001
	gridSearchConfig.putValue<double>("GridSearch.cGrid.Max", 1000.1);  //10000000000000050.1
	gridSearchConfig.putValue<double>("GridSearch.cGrid.LogStep", 10);
	gridSearchConfig.putValue<double>("GridSearch.gammaGrid.Min", 0.001);
	gridSearchConfig.putValue<double>("GridSearch.gammaGrid.Max", 1000.1);
	gridSearchConfig.putValue<double>("GridSearch.gammaGrid.LogStep", 10);
	gridSearchConfig.putValue<int>("GridSearch.SubsetSize", 0); //0 means full training set, otherwise random subset of T will be selected
	gridSearchConfig.putValue<int>("GridSearch.SubsetRepeats", 1); //number of times subsets will be tested (does not apply when using full T)

	auto info = (testApp::getInfoAboutDataset(folder + "\\train.csv"));

	if (info.size > 1000)
	{
		gridSearchConfig.putValue<int>("GridSearch.SubsetSize", 256);
		poly.putValue<int>("GridSearch.SubsetSize", 256);
		linear.putValue<int>("GridSearch.SubsetSize", 256);
	}
	
	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		//{"GridSearchNoFS_overfitted", full_rbf_overfitted},
		//{"GridSearchNoFS", genetic::DefaultGridSearchConfig::getDefault()},
		{"GridSearchNoFS", gridSearchConfig},
		//{"GridSearchNoFS_Poly", poly},
		//{"GridSearchNoFS_Linear", linear},
		//{"GridSearchNoFS_RBF_POLY_GLOBAL", rbf_poly},
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs1)
	{
		configManager.setupDataset(config.second, folder, config.first + "_" + config.second.getValue<std::string>("Svm.KernelType"), outputFolder);
		configManager.setMetric(config.second);
		//configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		configManager.setupStopCondition(config.second);
		config.second.putValue("Name", "GridSearchNoFS");
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1), 
			config.first + "_" + config.second.getValue<std::string>("Svm.KernelType"), fold);
	}
}

inline void gridSearchSubsets(std::string folder, uint32_t fold, std::string outputFolder)
{
	auto rbf_k2 = genetic::DefaultGridSearchConfig::getDefault();
	rbf_k2.putValue<int>("GridSearch.SubsetSize", 2);
	
	auto rbf_k8 = genetic::DefaultGridSearchConfig::getDefault();
	rbf_k8.putValue<int>("GridSearch.SubsetSize", 8);

	auto rbf_k32 = genetic::DefaultGridSearchConfig::getDefault();
	rbf_k32.putValue<int>("GridSearch.SubsetSize", 32);

	auto rbf_k128 = genetic::DefaultGridSearchConfig::getDefault();
	rbf_k128.putValue<int>("GridSearch.SubsetSize", 128);

	auto rbf_k512 = genetic::DefaultGridSearchConfig::getDefault();
	rbf_k512.putValue<int>("GridSearch.SubsetSize", 512);

	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		//{"GridSearchNoFS_k2", rbf_k2},
		{"GridSearchNoFS_k8", rbf_k8},
		{"GridSearchNoFS_k32", rbf_k32},
		//{"GridSearchNoFS_k128", rbf_k128},
		//{"GridSearchNoFS_k512", rbf_k512},
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs1)
	{
		configManager.setupDataset(config.second, folder, config.first + "_" + config.second.getValue<std::string>("Svm.KernelType"), outputFolder);
		configManager.setMetric(config.second);
		configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		configManager.setupStopCondition(config.second);
		config.second.putValue("Name", "GridSearchNoFS");
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1),
			config.first + "_" + config.second.getValue<std::string>("Svm.KernelType"), fold);
	}
}

inline void constRbfConfigs(std::string folder, uint32_t fold, std::string outputFolder)
{
	auto rbfTest = genetic::DefaultKernelEvolutionConfig::getDefault();
	rbfTest.putValue("Name", "ImplementationTest");
	

	std::vector<std::pair<std::string, platform::Subtree>> configs1 =
	{
		{"ImplementationTest_cpp", rbfTest},
	};

	testApp::ConfigManager configManager;
	for (auto& config : configs1)
	{
		configManager.setupDataset(config.second, folder, config.first + "_" + config.second.getValue<std::string>("Svm.KernelType"), outputFolder);
		configManager.setMetric(config.second);
		//configManager.setGridKernelInitialPopulationGeneration(config.second);
		configManager.setRandomNumberGenerators(config.second);
		configManager.setupStopCondition(config.second);
		//config.second.putValue("Name", "ImplementationTest");
		std::vector<std::string> elements = testApp::splitPath(folder);
		configManager.saveConfigToFileFolds(config.second, outputFolder + "\\" + *(elements.end() - 2) + "\\" + *(elements.end() - 1),
			config.first + "_" + config.second.getValue<std::string>("Svm.KernelType"), fold);
	}
}

inline void createAlgaConfigsRegression(std::string folder, uint32_t fold, const std::vector<uint32_t>& Kvalues)
{
	std::vector<std::pair<std::string, platform::Subtree>> configsAlga =
	{
		{"ALGA", genetic::DefaultAlgaConfig::getALGA_regression()},
	};
	testApp::ConfigManager configManager;
	for (auto& config : configsAlga)
	{
		for (auto K : Kvalues)
		{
			const auto algorithmName = config.first + "_" + std::to_string(K);

			//configManager.setupSuperPixelsDataset(config.second, folder, algorithmName);

			configManager.setMetricRegression(config.second);
			configManager.setGridKernelInitialPopulationGeneration(config.second);
			configManager.setRandomNumberGenerators(config.second);
			configManager.setK(config.second, K);
			configManager.setupStopCondition(config.second);
			configManager.saveConfigToFileFolds(config.second, folder, algorithmName + "_", fold);
		}
	}
}

inline void saveParametersForFold(std::map<uint32_t, KernelParams>& gridSearchResults, phd::svm::ISvm& resultModel, uint32_t fold)
{
	gridSearchResults.emplace(fold, KernelParams(resultModel.getC(), resultModel.getGamma()));
}


//#define SHORT_PATH

inline void runSpecified(std::filesystem::path pathToFold,
                         uint32_t foldNumber,
                         std::map<uint32_t, KernelParams>& gridSearchResults,
                         std::vector<std::string>& filters,
                         bool withFeatureSelection,
                         platform::Verbosity verbosity)
{
	try
	{
		genetic::SvmAlgorithmFactory fac;

		auto configs = testApp::getAllConfigFiles(pathToFold);

		const auto con111 = genetic::SvmWokrflowConfiguration(platform::Subtree(std::filesystem::path(configs[0])));
		std::vector<bool> featureMask;
		if (withFeatureSelection)
		{
			featureMask = runFeatureSelection(con111.trainingDataPath);
		}

		for (auto& file : configs)
		{
			auto config = platform::Subtree(std::filesystem::path(file));
			auto name = config.getValue<std::string>("Name");
			if (std::find(filters.begin(), filters.end(), name) == filters.end())
			{
				continue;
			}
			std::cout << file << "\n";


			
			auto basePath = config.getValue<std::string>("Svm.OutputFolderPath");

			//make short path
#ifdef SHORT_PATH
			basePath= basePath.substr(0, basePath.find_last_of("\\/"));
			basePath = basePath.substr(0, basePath.find_last_of("\\/"));
			basePath = basePath.substr(0, basePath.find_last_of("\\/"));
			basePath += "\\";
			config.putValue<std::string>("Svm.OutputFolderPath", basePath);
			auto outputfolderName = basePath;
#else

			
			auto configValue = config.getValue<std::string>("Svm.OutputFolderPath");
			auto outputfolderName = testApp::createOutputFolder(configValue);
			outputfolderName.push_back('\\');
			config.putValue<std::string>("Svm.OutputFolderPath", outputfolderName);
#endif
				testApp::saveRepositoryState(outputfolderName);

				const auto summary(outputfolderName + timeUtils::getTimestamp() + "_" + file.filename().string() + "_summary.txt");
				config.save(outputfolderName + "\\config.json");

				std::ofstream summaryFile(summary);
				std::vector<std::string> logFileNames;

				const auto con = genetic::SvmWokrflowConfiguration(config);
				std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
				if (withFeatureSelection)
				{
					ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath, featureMask);
				}
				else
				{
					ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath);
				}

				auto repeatNumber = 5;
				if (config.getValue<std::string>( "Name") == "GridSearch" ||
					config.getValue<std::string>("Name") == "GridSearchNoFS" ||
					config.getValue<std::string>("Name") == "ImplementationTest" /*|| config.getValue<std::string>("Name") == "Alma"*/)
				{
					repeatNumber = 1;
				}

				/*if (config.getValue<std::string>("Name") == "MemeticTrainingSetSelection")
				{
					auto variance_value = svmComponents::svmUtils::variance(ptrToLoader->getTraningSet());
					if (variance_value != 0)
					{
						auto gamma_value = 1.0 / (ptrToLoader->getTraningSet().getSample(0).size() * variance_value);
						config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Gamma", gamma_value);
					}
					else
					{
						auto gamma_value = 1.0;
						config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Gamma", gamma_value);
					}
					config.putValue<double>("Svm.MemeticTrainingSetSelection.Kernel.C", 1.0);
					LOG_F(INFO, "Final RBF params for MASVM: C=1, gamma=%f", config.getValue<double>("Svm.MemeticTrainingSetSelection.Kernel.Gamma"));
				}*/


				for (auto i = 0; i < repeatNumber; i++)
				{
					//Setting seed for experiment
					//testApp::ConfigManager cm;
					//cm.setSeedForRng(config, 1 + i);
					//cm.setSeedForRng(config, 256*4 + i);
					//cm.setSeedForRng(config, 256 + i);
					//
					//
			/*		std::cout << "SEEDS: \n";
					std::cout << config.getValue<int>("Svm.MemeticFeatureSetSelection.RandomNumberGenerator.Seed") << "\n";
					std::cout << config.getValue<int>("Svm.MemeticTrainingSetSelection.RandomNumberGenerator.Seed") << "\n";
					std::cout << config.getValue<int>("Svm.GeneticKernelEvolution.RandomNumberGenerator.Seed") << "\n";*/

	
					
					std::string  logfile(outputfolderName + "exp_" + std::to_string(i) + "_.log");
					if (verbosity == testApp::Verbosity::All)
					{
						loguru::add_file(logfile.c_str(), loguru::Append, loguru::Verbosity_MAX);
					}

					if (withFeatureSelection)
					{
						std::ofstream featureMaskForModel(outputfolderName + "featureMask_" + std::to_string(i) + "_.features");
						for (auto f : featureMask)
						{
							if (f)
							{
								featureMaskForModel << "1\n";
							}
							else
							{
								featureMaskForModel << "0\n";
							}
						}
					}
					
					std::cout << "Repeat:" << i << "\n";

					auto configFileName = file.stem().string();
					const auto logFilename = testApp::getLogFilename(foldNumber, i, configFileName);
					config.putValue<std::string>("Svm.TxtLogFilename", logFilename);
					auto al = fac.createAlgorightm(config, *ptrToLoader);
					//al->setC(C);
					const auto resultModel = al->run();

					if (verbosity == testApp::Verbosity::Standard || verbosity == testApp::Verbosity::All)
					{
						testApp::saveSvmModel(config, foldNumber, i, *resultModel, configFileName);
					}

					if (config.getValue<std::string>("Name") == "GridSearch" || config.getValue<std::string>("Name") == "GridSearchNoFS")
					{
						saveParametersForFold(gridSearchResults, *resultModel, foldNumber);
					}

					logFileNames.push_back(logFilename);


					if (verbosity == testApp::Verbosity::All && ptrToLoader->getTraningSet().hasGroups())
					{
						testApp::saveSvmGroupsResultsToFile(config, foldNumber, i, *resultModel, configFileName, *ptrToLoader);
					}

					if (verbosity == testApp::Verbosity::All)
					{
						//testApp::saveSvmResultsToFile(config, foldNumber, i, *resultModel, configFileName, *ptrToLoader); NOT NEEDED
						loguru::remove_callback(logfile.c_str());
					}
				}

				if (verbosity == testApp::Verbosity::Minimal)
				{
					testApp::createSummaryFile(config, summaryFile, logFileNames, verbosity);
				}
		}
	}
	catch (const std::exception& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
		std::cout << e.what() << "\n";
	}
}
