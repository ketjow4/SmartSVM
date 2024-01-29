#include "RunAlgorithm.h"



#include "libGeneticSvm/DefaultWorkflowConfigs.h"
#include "AppUtils/PythonFeatureSelection.h"

#include "Commons.h"
#include "Gecco2019.h"


void RunSingleAlgorithm(std::vector<std::string> dataFolders, std::vector<std::string> filters);

void RunESVMExperiments(std::vector<std::string> dataFolders);
void RunSESVMExperiments(std::vector<std::string> dataFolders);
void RunAKSVMExperiments(std::vector<std::string> dataFolders);
void RunDASVMExperiments(std::vector<std::string> dataFolders);
void RunDASVM_CE_FS_Experiments(std::vector<std::string> dataFolders, std::vector<std::string>& filters);
void RunDASVM_CE_Experiments(std::vector<std::string> dataFolders, std::vector<std::string>& filters);
void RunAlmaExperiments(std::vector<std::string> dataFolders);
void RunSESVMCorrectedExperiments(std::vector<std::string> dataFolders);

void RunAlgorithm(int argc, char* argv[], AlgorithmName algorithmToRun)
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	std::map<uint32_t, KernelParams> gridSearchResults;
	std::map<std::string, testApp::DatasetInfo> datasetInfos = getDatasetInformations(config);

	//Config generation--------------------------------------------------------------------------------
	auto dataFolders = testApp::listDirectories(config.datafolder);

	auto outputFolders = testApp::listDirectories(outputResultsDir);
	int i = 0;
	
	for (auto& folder : dataFolders)
	{
		//const std::vector<uint32_t>& Kvalues = datasetInfos[folder].kValues;
		auto allfoldFolders = testApp::listDirectories(folder);

		if(algorithmToRun == AlgorithmName::MASVM || algorithmToRun == AlgorithmName::GASVM)
		{
			if (outputFolders.size() != 0)
			{
				loadGridSearchParams(gridSearchResults, outputFolders[i], "GridSearchNoFS");
				i++;
			}
			else if (gridSearchResults.empty())
			{
				LOG_F(WARNING, "No grid search results, RBF params will be default as in sci-kit for MASVM and GASVM (C=1, gamma=scale)");
				LOG_F(WARNING, "These parameters are caclucated in code and are not saved into json config file (final params are in svm xml file)");
				gridSearchResults[1] = KernelParams(1, 1);
				gridSearchResults[2] = KernelParams(1, 1);
				gridSearchResults[3] = KernelParams(1, 1);
				gridSearchResults[4] = KernelParams(1, 1);
				gridSearchResults[5] = KernelParams(1, 1);
			}
			else
			{
				LOG_F(WARNING, "No grid search results, RBF params will be default as in sci-kit for MASVM and GASVM (C=1, gamma=scale)");
				LOG_F(WARNING, "These parameters are caclucated in code and are not saved into json config file (final params are in svm xml file)");
				gridSearchResults[1] = KernelParams(1, 1);
				gridSearchResults[2] = KernelParams(1, 1);
				gridSearchResults[3] = KernelParams(1, 1);
				gridSearchResults[4] = KernelParams(1, 1);
				gridSearchResults[5] = KernelParams(1, 1);
			}

		}
		for (auto& foldFolder : allfoldFolders)
		{
				switch (algorithmToRun)
				{
				case AlgorithmName::GridSearch:
				{
					gridSearchWithOutFeatures(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
					break;
				}
				case AlgorithmName::GridSearchWithFetureSelection:
				{
					throw std::exception("Grid Search With Feature Selection not implemented yet");
					//TODO later, not needed for now
					//break;
				}
				case AlgorithmName::GASVM:
				{
					auto foldNumber = getFoldNumberFromFolder(foldFolder);
					createGasvmConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), datasetInfos[folder].kValues, outputResultsDir.string(),
					                   gridSearchResults.at(foldNumber));
					break;
				}
				case AlgorithmName::MASVM:
				{
					auto foldNumber = getFoldNumberFromFolder(foldFolder);
					createMasvmConfiguration(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), gridSearchResults.at(foldNumber));
					break;
				}
				case AlgorithmName::ALGA:
				{
					createAlgaConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), datasetInfos[folder].kValues, outputResultsDir.string());
					break;
				}
				case AlgorithmName::ALMA:
				{
					createAlmaConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					break;
				}
				case AlgorithmName::FSALMA:
				{
					createFSAlmaConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					break;
				}
				case AlgorithmName::ESVM:
				{
					createESVMConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					break;
				}
				case AlgorithmName::SE_SVM:
				{
					createSESVMConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					break;
				}
				case AlgorithmName::AKSVM:
				{
					createAKSVMConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					break;
				}
				case AlgorithmName::DA_SVM_MIN:
				case AlgorithmName::DA_SVM_AVG:
				case AlgorithmName::DA_SVM_MAX:
				case AlgorithmName::DA_SVM_SUM:
				case AlgorithmName::DA_SVM_NO_T:
				{
					LOG_F(WARNING, "Only DASVM_MAX is supported");
					createDASVMConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					//need to specify more versions
					break;
				}
				case AlgorithmName::DA_SVM_CE_FS_ONE:
				{
					createDASVM_Variants_Configs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder], AlgorithmName::DA_SVM_CE_FS_ONE);
					break;
				}
				case AlgorithmName::DA_SVM_CE_FS_MAX:
				{
					createDASVM_Variants_Configs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder], AlgorithmName::DA_SVM_CE_FS_MAX);
					break;
				}
				case AlgorithmName::DA_SVM_CE_FS_NO_T:
				{
					createDASVM_Variants_Configs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder], AlgorithmName::DA_SVM_CE_FS_NO_T);
					break;
				}
				case AlgorithmName::DA_SVM_CE_ONE:
				{
					createDASVM_Variants_Configs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder], AlgorithmName::DA_SVM_CE_ONE);
					break;
				}
				case AlgorithmName::DA_SVM_CE_MAX:
				{
					createDASVM_Variants_Configs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder], AlgorithmName::DA_SVM_CE_MAX);
					break;
				}
				case AlgorithmName::DA_SVM_CE_NO_T:
				{
					createDASVM_Variants_Configs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder], AlgorithmName::DA_SVM_CE_NO_T);
					break;
				}
				case AlgorithmName::SE_SVM_CORRECTED:
				{
					createSESVMCorrectedConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
					break;
				}
				default: ;
				}
			
		}
	}
	
	dataFolders = testApp::listDirectories(outputResultsDir);

	switch (algorithmToRun)
	{
	case AlgorithmName::GridSearch:
	{
		RunGridSearch(gridSearchResults, dataFolders);
		break;
	}
	case AlgorithmName::GridSearchWithFetureSelection: break;
	case AlgorithmName::GASVM:
	{
		RunGasvmExperiments(dataFolders);
		break;
	}
	case AlgorithmName::MASVM:
	{
		RunMasvmExperiments(dataFolders);
		break;
	}
		
	case AlgorithmName::ALGA:
	{
		RunAlgaExperiments(dataFolders);
		break;
	}
	case AlgorithmName::ALMA:
	{
		RunAlmaExperiments(dataFolders);
		break;
	}
	case AlgorithmName::FSALMA:
	{
		RunFSALMAExperiments(dataFolders);
		break;
	}
	case AlgorithmName::ESVM:
	{
		RunESVMExperiments(dataFolders);
		break;
	}
	case AlgorithmName::SE_SVM:
	{
		RunSESVMExperiments(dataFolders);
		break;
	}
	case AlgorithmName::AKSVM:
	{
		RunAKSVMExperiments(dataFolders);
		break;
	}
	case AlgorithmName::DA_SVM_MAX:
	{
		RunDASVMExperiments(dataFolders);
		break;
	}
	case AlgorithmName::DA_SVM_CE_ONE:
	{
		std::vector<std::string> variant{ "RbfLinearCoevolutionCEONE" };
		RunDASVM_CE_Experiments(dataFolders, variant);
		break;
	}
	case AlgorithmName::DA_SVM_CE_MAX:
	{
		std::vector<std::string> variant{ "RbfLinearCoevolutionCEMAX" };
		RunDASVM_CE_Experiments(dataFolders, variant);
		break;
	}
	case AlgorithmName::DA_SVM_CE_NO_T:
	{
		std::vector<std::string> variant{ "RbfLinearCoevolutionCENOT" };
		RunDASVM_CE_Experiments(dataFolders, variant);
		break;
	}
	case AlgorithmName::DA_SVM_CE_FS_ONE :
	{
		std::vector<std::string> variant{ "RbfLinearCoevolutionFSCEONE" };
		RunDASVM_CE_FS_Experiments(dataFolders, variant);
		break;
	}
	case AlgorithmName::DA_SVM_CE_FS_MAX:
	{
		std::vector<std::string> variant{ "RbfLinearCoevolutionFSCEMAX" };
		RunDASVM_CE_FS_Experiments(dataFolders, variant);
		break;
	}
	case AlgorithmName::DA_SVM_CE_FS_NO_T: 
	{
		std::vector<std::string> variant{ "RbfLinearCoevolutionFSCENOT" };
		RunDASVM_CE_FS_Experiments(dataFolders, variant);
		break;
	}
	case AlgorithmName::SE_SVM_CORRECTED:
	{
		RunSESVMCorrectedExperiments(dataFolders);
		break;
	}
	default: ;
	}
	

	return;
}

std::map<std::string, testApp::DatasetInfo> getDatasetInformations(const testApp::configTestApp& config)
{
	std::map<std::string, testApp::DatasetInfo> datasetInfos;
	auto dataFolders2 = testApp::listDirectories(config.datafolder);
	for (auto& folder : dataFolders2)
	{
		if(std::filesystem::exists(folder + "\\1\\train.csv"))
		{
			datasetInfos[folder] = (testApp::getInfoAboutDataset(folder + "\\1\\train.csv"));
		}
		else if (std::filesystem::exists(folder + "\\1\\train.groups"))
		{
			datasetInfos[folder] = (testApp::getInfoAboutDataset(folder + "\\1\\train.groups"));
		}
	}

	return datasetInfos;
}

void RunRandomSearchExperiments(std::vector<std::string> dataFolders)
{
	std::map<uint32_t, KernelParams> notUsedOne;
	
	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			std::vector<std::string> filters{"RandomSearch", "RandomSearchInitPop", "RandomSearchEvoHelp"};

			consolidateExperiment(foldFolder, foldNumber, notUsedOne, filters, false);
			//runSpecified(foldFolder, foldNumber, gridSearchResults, filters, false);
		}
	}
}

void RunAlgaExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "Alga" });
}

void RunAlmaExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "Alma" });
}

void RunAKSVMExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "SequentialGamma" });
}

void RunDASVMExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "SequentialGamma", "RbfLinear", "RbfLinearCoevolution" });
}

void RunDASVM_CE_Experiments(std::vector<std::string> dataFolders, std::vector<std::string>& filters)
{
	RunSingleAlgorithm(dataFolders, filters);
}

void RunDASVM_CE_FS_Experiments(std::vector<std::string> dataFolders, std::vector<std::string>& filters)
{
	RunSingleAlgorithmFS(dataFolders, filters);
}


void RunESVMExperiments(std::vector<std::string> dataFolders)
{
	//"MemeticTrainingSetSelection", "FeatureSetSelection", "TF", "FT"
	RunSingleAlgorithm(dataFolders, { "KTF" });
}

void RunSESVMExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "SSVM" });
}

void RunSESVMCorrectedExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "SESVM_Corrected" });
}

void RunMasvmExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "MemeticTrainingSetSelection" });
}

void RunGasvmExperiments(std::vector<std::string> dataFolders)
{
	RunSingleAlgorithm(dataFolders, { "GaSvm" });
}

void RunFSALMAExperiments( std::vector<std::string> dataFolders)
{
	std::map<uint32_t, KernelParams> notUsedOne;
	
	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			std::vector<std::string> filters{"FSAlma"}; 

			runSpecified(foldFolder, foldNumber, notUsedOne, filters, true);
		}
	}
}

void RunFSALMA(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	std::map<std::string, testApp::DatasetInfo> datasetInfos = getDatasetInformations(config);

	//Config generation--------------------------------------------------------------------------------
	auto dataFolders = testApp::listDirectories(config.datafolder);
	for (auto& folder : dataFolders)
	{
		const std::vector<uint32_t>& Kvalues = datasetInfos[folder].kValues;
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			createFSAlmaConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[foldFolder]);
		}
	}
	dataFolders = testApp::listDirectories(outputResultsDir);
	RunFSALMAExperiments(dataFolders);
}

void RunGridSearch(std::map<uint32_t, KernelParams>& gridSearchResults, std::vector<std::string> dataFolders)
{
	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			std::vector<std::string> filters{"GridSearchNoFS"};

			runSpecified(foldFolder, foldNumber, gridSearchResults, filters, false);
		}
		saveGridSearchResultsToFile(gridSearchResults, datasetFolder, "GridSearchNoFS");
	}
}

void RunGsNoFeatureSelection(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	std::map<uint32_t, KernelParams> gridSearchResults;
	std::map<std::string, testApp::DatasetInfo> datasetInfos = getDatasetInformations(config);

	//Config generation--------------------------------------------------------------------------------
	auto dataFolders = testApp::listDirectories(config.datafolder);
	for (auto& folder : dataFolders)
	{
		const std::vector<uint32_t>& Kvalues = datasetInfos[folder].kValues;
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			//createConfigs1(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[foldFolder]);
			gridSearchWithOutFeatures(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
			//createAlgaConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), Kvalues);
		}
	}

	dataFolders = testApp::listDirectories(outputResultsDir);
	RunGridSearch(gridSearchResults, dataFolders);

	//end
	return;
}

void RunGridSearchWithFeatureSelection(std::map<uint32_t, KernelParams>& gridSearchResults, std::vector<std::string> dataFolders)
{
	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			std::vector<std::string> filters{"GridSearch"}; //@wdudzik note that alma == alga   

			consolidateExperiment(foldFolder, foldNumber, gridSearchResults, filters, true);
			//runSpecified(foldFolder, foldNumber, gridSearchResults, filters, false);
		}
		saveGridSearchResultsToFile(gridSearchResults, datasetFolder, "GridSearchFS");
		//gridSearchResults.clear();
	}
}



void RunSsvmGecco2019(std::vector<std::string> dataFolders)
{
	std::map<uint32_t, KernelParams> notUsedOne;

	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			std::vector<std::string> filters{"SSVM", "KTF"}; //@wdudzik note that alma == alga   

			//tutaj uruchamiamy z Mutual Infromation
			//wersja Gecco 2019
			consolidateExperiment(foldFolder, foldNumber, notUsedOne, filters, false);

			//wersja zwykla
			//runSpecified(foldFolder, foldNumber, gridSearchResults, filters, false);
		}
	}
}

void RunEnsembleExperiments(std::vector<std::string> dataFolders)
{
	std::map<uint32_t, KernelParams> notUsedOne;

	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);
			std::vector<std::string> filters{"Ensemble", "EnsembleTree", "BigSetsEnsemble"};

			runSpecified(foldFolder, foldNumber, notUsedOne, filters, false);
		}
	}
}


void RunSingleAlgorithm(std::vector<std::string> dataFolders, std::vector<std::string> filters)
{
	std::map<uint32_t, KernelParams> notUsedOne;

	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);
			runSpecified(foldFolder, foldNumber, notUsedOne, filters, false);
		}
	}
}

void RunSingleAlgorithmFS(std::vector<std::string> dataFolders, std::vector<std::string> filters)
{
	std::map<uint32_t, KernelParams> notUsedOne;

	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);
			runSpecified(foldFolder, foldNumber, notUsedOne, filters, true);
		}
	}
}