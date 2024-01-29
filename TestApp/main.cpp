#include <algorithm>
#include <string>
#include "ConfigParser.h"
//#include "AppUtils/AppUtils.h"
#include "ConfigGeneration.h"

#include "libGeneticSvm/DefaultWorkflowConfigs.h"
//#include "AppUtils/PythonFeatureSelection.h"

#include "Commons.h"
#include "Gecco2019.h"
#include "CustomKernelExperiments.h"
#include "libSvmComponents/DataNormalization.h"
#include "libSvmComponents/CustomWidthGauss.h"

#include <libPlatform/loguru.cpp>
#include <libPlatform/loguru.hpp>

#include "libSvmComponents/SvmAucMetric.h"
#include "ManualMetricTests.h"
#include "libGeneticSvm/GridSearchWorkflow.h"
#include "SvmLib/EnsembleListSvm.h"
#include "libSvmComponents/RbfKernel.h"
#include "libSvmComponents/SvmAccuracyMetric.h"
#include "libSvmComponents/SvmAucMetric.h"
#include "ManualRbfLiniear.h"
#include "ReRuns.h"
#include "RunAlgorithm.h"
#include "LastRegionsScores.h"

// #include <pybind11/embed.h>
// #include <pybind11/numpy.h>
#include <stdlib.h>


#include "libGeneticSvm/SvmEnsembleHelper.h"
#include "SvmLib/VotingEnsemble.h"

#include <regex>
#include "libDataset/CsvReader.h"

void enhancedTrainingSetAndValidationSetExperiments(int argc, char* argv[])
{
	//parse cmd arguments
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	std::map<uint32_t, KernelParams> gridSearchResults;
	std::map<std::string, testApp::DatasetInfo> datasetInfos = getDatasetInformations(config);

	auto dataFolders = testApp::listDirectories(config.datafolder);

	//prepare configuration for grid search algortihm for finding the best C, gamma parameters for RBF kernel
	for (auto& folder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			gridSearchWithOutFeatures(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
		}
	}

	//RUN grid search to obtain values for kernel hyperparameters, these results are saved to files
	dataFolders = testApp::listDirectories(outputResultsDir);
	//RunGridSearch(gridSearchResults, dataFolders);

	dataFolders = testApp::listDirectories(config.datafolder);
	//create proper configurations - setup the enhancement of training set for MASVM
	auto enhanced_training_set = genetic::DefaultMemeticConfig::getDefault();
	enhanced_training_set.putValue<bool>("Svm.MemeticTrainingSetSelection.EnhanceTrainingSet", true);
	enhanced_training_set.putValue<std::string>("Svm.MemeticTrainingSetSelection.Crossover.Name", "EnhanceTrainingSet");
	enhanced_training_set.putValue<std::string>("Svm.MemeticTrainingSetSelection.Generation.Name", "EnhanceTrainingSet");
	enhanced_training_set.putValue<std::string>("Svm.MemeticTrainingSetSelection.Mutation.Name", "EnhanceTrainingSet");

	//setup selection of validation set for MASVM
	auto enhanced_validation_set = genetic::DefaultMemeticConfig::getDefault();
	enhanced_validation_set.putValue<std::string>("Svm.MemeticTrainingSetSelection.Validation.Name", "Subset");
	enhanced_validation_set.putValue<std::string>("Svm.MemeticTrainingSetSelection.Validation.Method", "Dummy");

	//GASVM setup with enhanced training
	auto enhanced_training_set_gasvm = genetic::DefaultGaSvmConfig::getDefault();
	enhanced_training_set_gasvm.putValue<bool>("Svm.GaSvm.EnhanceTrainingSet", true);
	enhanced_training_set_gasvm.putValue<std::string>("Svm.GaSvm.Crossover.Name", "EnhanceTrainingSet");
	enhanced_training_set_gasvm.putValue<std::string>("Svm.GaSvm.Generation.Name", "EnhanceTrainingSet");
	enhanced_training_set_gasvm.putValue<std::string>("Svm.GaSvm.Mutation.Name", "EnhanceTrainingSet");

	//GASVM setup with  selection of validation set 
	auto enhanced_validation_set_gasvm = genetic::DefaultGaSvmConfig::getDefault();
	enhanced_validation_set_gasvm.putValue<std::string>("Svm.GaSvm.Validation.Name", "Subset");
	enhanced_validation_set_gasvm.putValue<std::string>("Svm.GaSvm.Validation.Method", "Dummy");

	//creation of experiment name to config pairs
	std::vector<std::pair<std::string, platform::Subtree>> configs =
	{
		//{"MASVM_basic", genetic::DefaultMemeticConfig::getDefault()},
		//{"MASVM_enhanced_training_set", enhanced_training_set},
		//{"MASVM_enhanced_validation_set", enhanced_validation_set},

		//{"GASVM_basic", genetic::DefaultGaSvmConfig::getDefault()},
		//{"GASVM_enhanced_training_set", enhanced_training_set_gasvm},
		{"GASVM_enhanced_validation_set", enhanced_validation_set_gasvm},
	};

	//saving the configuration into *.json files
	for (auto& folder : dataFolders)
	{
		auto kValuesforGasvm = datasetInfos[folder].kValues;
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);
			//processConfigurationsAndSave(configs, foldFolder, foldNumber, outputResultsDir.string(), gridSearchResults.at(foldNumber));
			processConfigurationsAndSave(configs, foldFolder, foldNumber, outputResultsDir.string(), KernelParams(0.0001, 100), datasetInfos[folder]);
		}
	}

	//run MASVM & GASVM
	dataFolders = testApp::listDirectories(outputResultsDir);
	RunGasvmExperiments(dataFolders);
	RunMasvmExperiments(dataFolders);
}

void newMain(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	//outputResultsDir = config.datafolder;

	//fold number mapped to kernel paramas
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
			createConfigs1(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
			//gridSearchWithOutFeatures(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
			//createAlgaConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), Kvalues);
		}
	}

	dataFolders = testApp::listDirectories(outputResultsDir);
	RunAlgaExperiments(dataFolders);

	RunSsvmGecco2019(dataFolders);
	/*
	RunSsvmGecco2019(gridSearchResults, dataFolders);
	RunGridSearchWithFeatureSelection(gridSearchResults, dataFolders);
	RunRandomSearchExperiments(gridSearchResults, dataFolders);*/
	//svmComponents::DataNormalization::useDefinedMinMax(0, 500);
	//RunGridSearch(gridSearchResults, dataFolders);

	//FT, TF Masvm, FeatureSet
	dataFolders = testApp::listDirectories(config.datafolder);
	auto outputFolders = testApp::listDirectories(outputResultsDir);
	int i = 0;
	for (auto& datasetFolder : dataFolders)
	{
		/*  auto numberOfFeatures = datasetInfos[datasetFolder].numberOfFeatures; 
		  numberOfFeatures = datasetInfos[datasetFolder].numberOfFeatures < datasetInfos[datasetFolder].size ? datasetInfos[datasetFolder].numberOfFeatures : 32;*/
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		loadGridSearchParams(gridSearchResults, outputFolders[i], "GridSearchNoFS");
		i++;
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			//this 8 is the same as setting for multiple and sequential gamma
			//createMasvmConfiguration(foldFolder, 8, foldNumber, outputResultsDir.string(), gridSearchResults.at(foldNumber));
			createMasvmConfiguration(foldFolder, foldNumber, outputResultsDir.string(), KernelParams(1, 1));
		}
	}

	//run MASVM & GASVM
	dataFolders = testApp::listDirectories(outputResultsDir);
	//RunMasvmAndGasvmExperiments(gridSearchResults, dataFolders);

	//end
	return;
}

void GridSearchLinearRbfFull(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	//fold number mapped to kernel paramas
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
			gridSearchWithOutFeatures(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
		}
	}

	dataFolders = testApp::listDirectories(outputResultsDir);

	RunGridSearch(gridSearchResults, dataFolders);

	return;
}


void GridSearchRbfSubsetsTest(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	//outputResultsDir = config.datafolder;

	//fold number mapped to kernel paramas
	std::map<uint32_t, KernelParams> gridSearchResults;
	std::map<std::string, testApp::DatasetInfo> datasetInfos = getDatasetInformations(config);

	//Config generation--------------------------------------------------------------------------------
	auto dataFolders = testApp::listDirectories(config.datafolder);
	//KTF , GridSearch,  Alma, Alga
	for (auto& folder : dataFolders)
	{
		const std::vector<uint32_t>& Kvalues = datasetInfos[folder].kValues;
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			gridSearchSubsets(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
		}
	}

	dataFolders = testApp::listDirectories(outputResultsDir);
	//svmComponents::DataNormalization::useDefinedMinMax(0, 500);
	RunGridSearch(gridSearchResults, dataFolders);

	return;
}

void ConstRbfTest(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	//outputResultsDir = config.datafolder;

	//fold number mapped to kernel paramas
	std::map<uint32_t, KernelParams> gridSearchResults;
	std::map<std::string, testApp::DatasetInfo> datasetInfos = getDatasetInformations(config);

	//Config generation--------------------------------------------------------------------------------
	auto dataFolders = testApp::listDirectories(config.datafolder);
	//KTF , GridSearch,  Alma, Alga
	for (auto& folder : dataFolders)
	{
		const std::vector<uint32_t>& Kvalues = datasetInfos[folder].kValues;
		auto allfoldFolders = testApp::listDirectories(folder);
		for (auto& foldFolder : allfoldFolders)
		{
			constRbfConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string());
		}
	}

	dataFolders = testApp::listDirectories(outputResultsDir);
	for (auto& datasetFolder : dataFolders)
	{
		auto allfoldFolders = testApp::listDirectories(datasetFolder);
		for (auto& foldFolder : allfoldFolders)
		{
			auto foldNumber = getFoldNumberFromFolder(foldFolder);

			std::vector<std::string> filters{"ImplementationTest"};

			runSpecified(foldFolder, foldNumber, gridSearchResults, filters, false);
		}
	}

	return;
}

void newEnsemble(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	auto outputResultsDir = std::filesystem::path(config.outputFolder);

	//fold number mapped to kernel paramas
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
			createEnsembleConfigs(foldFolder, getFoldNumberFromFolder(foldFolder), outputResultsDir.string(), datasetInfos[folder]);
		}
	}

	dataFolders = testApp::listDirectories(outputResultsDir);
	RunEnsembleExperiments(dataFolders);

	//end
	return;
}




class TestClassificationsSchemes
{
public:
	TestClassificationsSchemes(int argc, char* argv[])
	{
		m_config = testApp::parseCommandLineArguments(argc, argv);
	}

	std::string getJSON(std::filesystem::path folderPath)
	{
		std::vector<std::string> configFiles;
		for (auto& file : std::filesystem::recursive_directory_iterator(folderPath))
		{
			if (file.path().extension().string() == ".json")
			{
				configFiles.push_back(file.path().string());
			}
		}
		if(configFiles.size() != 1)
		{
			LOG_F(ERROR, std::string("Too many json files found in output folder " + folderPath.string()).c_str());
			throw std::exception(std::string("Too many json files found in output folder " + folderPath.string()).c_str());
		}
		
		return configFiles[0];
	}


	bool hasEnding(std::string const& fullString, std::string const& ending) {
		if (fullString.length() >= ending.length()) {
			return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
		}
		else {
			return false;
		}
	}
	
	std::vector<std::string> getWeights(std::filesystem::path folderPath)
	{
		std::vector<std::string> weightFiles;
		for (auto& file : std::filesystem::recursive_directory_iterator(folderPath))
		{
			if (hasEnding(file.path().string(), "__weights.txt"))
			{
				weightFiles.push_back(file.path().string());
			}
		}
		/*if (weightFiles.size() != 1)
		{
			LOG_F(ERROR, std::string("Too many weights files found in output folder " + folderPath.string()).c_str());
			throw std::exception(std::string("Too many weights files found in output folder " + folderPath.string()).c_str());
		}*/

		return weightFiles;
	}

	auto evaluateForDataset(phd::svm::VotingEnsemble& svmEnsemble, const dataset::Dataset<std::vector<float>, float>& dataset)
	{
		std::array<std::array<uint32_t, 2>, 2> matrix = { 0 };
		std::vector<std::int64_t> wrongOnesIds(dataset.size(), -1);

		auto samples = dataset.getSamples();
		auto labels = dataset.getLabels();
		int uncertainNumber = 0;

#pragma omp parallel for
		for (long long i = 0; i < static_cast<long long>(dataset.size()); i++)
		{
			//auto class_score = svmEnsemble.classifyWithCertainty(samples[i]);
			auto class_score = svmEnsemble.classifyNodeWeights(samples[i]);
			if (class_score != -100)
			{
			#pragma omp critical
				++matrix[static_cast<int>(class_score)][static_cast<int>(labels[i])];
			}
			else
			{
			#pragma omp critical
					uncertainNumber++;
			}
		}

		//svmComponents::ConfusionMatrix cm(svmEnsemble, dataset);
		//std::cout << cm << "\n";
		return std::make_pair(matrix, uncertainNumber);
	}


	std::vector<std::filesystem::path> getAllSvms(std::filesystem::path folderPath)
	{
		std::vector<std::string> svms;
		for (auto& file : std::filesystem::recursive_directory_iterator(folderPath))
		{
			if (file.path().extension().string() == ".xml")
			{
				svms.push_back(file.path().string());
			}
		}
		std::sort(svms.begin(), svms.end(), compareNat);
		return std::vector<std::filesystem::path>(svms.begin(), svms.end());
	}

	std::pair<std::string, std::string> process_and_save_regions(std::string& base_path, const dataset::Dataset<std::vector<float>, float>& dataset,
	                                                             phd::svm::VotingEnsemble& svmEnsemble, int repeatNumber, std::string datasetName)
	{
		auto path = datasetName + "_certain_" + std::to_string(repeatNumber) + ".csv";
		auto path_uncertain = datasetName + "_uncertain_" + std::to_string(repeatNumber) + ".csv";

		auto samples = dataset.getSamples();
		auto labels = dataset.getLabels();

		std::ofstream certain(base_path + "\\" + path);
		std::ofstream uncertain(base_path + "\\" + path_uncertain);

#pragma omp parallel for
		for (auto j = 0; j < static_cast<int>(dataset.size()); ++j)
		{
			auto result = svmEnsemble.classifyWithCertainty(samples[j]); 
			
			if (result != -100)
			{
			#pragma omp critical
				{
				//sample to csv  https://stackoverflow.com/a/8581865/8855783 there will be trailing ',' so we can add label safely
				std::copy(samples[j].begin(), samples[j].end(),
					std::ostream_iterator<float>(certain, ","));
		
				certain << labels[j] << "\n";
				}
			}
			else
			{
			#pragma omp critical
				{
					//sample to csv  https://stackoverflow.com/a/8581865/8855783 there will be trailing ',' so we can add label safely
					std::copy(samples[j].begin(), samples[j].end(),
						std::ostream_iterator<float>(uncertain, ","));

					uncertain << labels[j] << "\n";
				}
			}
		}
		certain.close();
		uncertain.close();

		return  { base_path + "\\" + path , base_path + "\\" + path_uncertain };
	}



	std::pair<std::string, std::string> process_and_save_regions(std::string& base_path, const dataset::Dataset<std::vector<float>, float>& dataset,
		phd::svm::EnsembleListSvm& svmEnsemble, int repeatNumber, std::string datasetName)
	{
		auto path = datasetName + "_certain_" + std::to_string(repeatNumber) + ".csv";
		auto path_uncertain = datasetName + "_uncertain_" + std::to_string(repeatNumber) + ".csv";

		auto samples = dataset.getSamples();
		auto labels = dataset.getLabels();

		std::ofstream certain(base_path + "\\" + path);
		std::ofstream uncertain(base_path + "\\" + path_uncertain);

#pragma omp parallel for
		for (auto j = 0; j < static_cast<int>(dataset.size()); ++j)
		{
			auto result = svmEnsemble.classifyWithCertainty(samples[j]);

			if (result != -100)
			{
#pragma omp critical
				{
					//sample to csv  https://stackoverflow.com/a/8581865/8855783 there will be trailing ',' so we can add label safely
					std::copy(samples[j].begin(), samples[j].end(),
						std::ostream_iterator<float>(certain, ","));

					certain << labels[j] << "\n";
				}
			}
			else
			{
#pragma omp critical
				{
					//sample to csv  https://stackoverflow.com/a/8581865/8855783 there will be trailing ',' so we can add label safely
					std::copy(samples[j].begin(), samples[j].end(),
						std::ostream_iterator<float>(uncertain, ","));

					uncertain << labels[j] << "\n";
				}
			}
		}
		certain.close();
		uncertain.close();

		return  { base_path + "\\" + path , base_path + "\\" + path_uncertain };
	}
	
	inline void handleUncertain(std::ofstream& output, const dataset::Dataset<std::vector<float>, float>& data, svmComponents::BaseSvmChromosome individual)
	{
		svmComponents::SvmAccuracyMetric acc;

		if (!data.empty())
		{
			auto resultuncertainTest = acc.calculateMetric(individual, data, true, true);
			output << resultuncertainTest.m_confusionMatrix.value() << "\t";

			//std::cout << "Uncertain:" << resultuncertainTest.m_confusionMatrix << "\n";
		}
		else
		{
			output << svmComponents::ConfusionMatrix(0, 0, 0, 0) << "\t";
		}
	}

	inline void handleCertainAndUncertainSeSVM(std::ofstream& output,
	                                           const dataset::Dataset<std::vector<float>, float>& certainData,
	                                           svmComponents::BaseSvmChromosome individual,
	                                           svmComponents::BaseSvmChromosome SESVM,
	                                           const dataset::Dataset<std::vector<float>, float>& uncertainData)
	{
		svmComponents::SvmAccuracyMetric acc;

		svmComponents::Metric certain;
		svmComponents::Metric uncertain;

		if (!certainData.empty())
		{
			certain = acc.calculateMetric(individual, certainData, true, true);
			//output << resultuncertainTest.m_confusionMatrix << "\t";

			//std::cout << "Certain:" << resultuncertainTest.m_confusionMatrix << "\n";
		}
		if (!uncertainData.empty())
		{
			uncertain = acc.calculateMetric(SESVM, certainData, true, true);
			//output << resultuncertainTest.m_confusionMatrix << "\t";

			//std::cout << "Certain:" << resultuncertainTest.m_confusionMatrix << "\n";
		}
		else
		{
			uncertain = svmComponents::Metric(0,svmComponents::ConfusionMatrix(0, 0, 0, 0));
		}


	/*	ConfusionMatrix(uint32_t truePositive,
			uint32_t trueNegative,
			uint32_t falsePositive,
			uint32_t falseNegative);*/


		svmComponents::ConfusionMatrix allmatrix(certain.m_confusionMatrix.value().truePositive() + uncertain.m_confusionMatrix.value().truePositive(),
										certain.m_confusionMatrix.value().trueNegative() + uncertain.m_confusionMatrix.value().trueNegative(),
										certain.m_confusionMatrix.value().falsePositive() + uncertain.m_confusionMatrix.value().falsePositive(),
										certain.m_confusionMatrix.value().falseNegative() + uncertain.m_confusionMatrix.value().falseNegative());
	
		output << allmatrix << "\t";

		/*else
		{
			output << svmComponents::ConfusionMatrix(0, 0, 0, 0) << "\t";
		}*/

	}


	inline void handleCertain(std::ofstream& output, const dataset::Dataset<std::vector<float>, float>& data, svmComponents::BaseSvmChromosome individual)
	{
		svmComponents::SvmAccuracyMetric acc;

		if (!data.empty())
		{
			auto resultuncertainTest = acc.calculateMetric(individual, data, true, true);
			output << resultuncertainTest.m_confusionMatrix.value() << "\t";

			//std::cout << "Certain:" << resultuncertainTest.m_confusionMatrix << "\n";
		}
		else
		{
			output << svmComponents::ConfusionMatrix(0, 0, 0, 0) << "\t";
		}
	}
	
	void resulstForSingleRunRegions(std::filesystem::path fold, std::filesystem::path algorithm, std::string algorithmPath, std::vector<std::string> /*allCSVs*/,
		std::string classification_type, bool fiftyPercent, std::shared_ptr<phd::svm::libSvmImplementation> SESVM)
	{
		auto configFile = getJSON(algorithmPath);

		platform::Subtree config{std::filesystem::path(configFile)};

		const auto con = genetic::SvmWokrflowConfiguration(config);
		std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
		auto normalize = true;
		auto resample = false;
		ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath, normalize, resample);

		auto algorithmName = platform::stringUtils::splitString(algorithm.stem().string(), "__")[0];


		if (algorithmName.find("BigSetsEnsemble") == std::string::npos)
			return;
		
		auto svmPath = getAllSvms(algorithmPath);

		auto pattern = std::regex("(\\d+)fold");
		std::map<int, std::vector<std::filesystem::path>> repeats;
		for (auto p : svmPath)
		{
			std::smatch sm;
			const auto s = p.string();
			std::regex_search(s, sm, pattern);
			repeats[std::stoi(sm.str().substr(0, sm.str().size() - 4))].emplace_back(p);
		}


		auto weightsPath = getWeights(algorithmPath);

		std::cout << algorithmPath << "\n";
		std::cout << algorithmName << "\n";


		std::string variantName = classification_type;
		variantName = "SESVM_LAST";
		
		if (fiftyPercent)
		{
			variantName += "_50_percent_certain_";
		}

		algorithmName = "BigEnsembleOldScheme_" + variantName;
		//phd::svm::VotingEnsemble svmEnsemble(svmPath, weightsPath);
		std::ofstream output(fold.string() + "\\" + algorithmName + "_LastNode.txt");

		for (auto i = 0; i < repeats.size(); ++i)
		{
			auto weigthMock = "file.txt";

			std::shared_ptr<phd::svm::VotingEnsemble> svmEnsemble  = std::make_shared<phd::svm::VotingEnsemble>(repeats[i], weightsPath[i]); //phd::svm::VotingEnsemble svmEnsemble(repeats[i], weightsPath[i]);
			//std::shared_ptr<phd::svm::VotingEnsemble> svmEnsemble  = std::make_shared<phd::svm::VotingEnsemble>(repeats[i], weightsPath[i], classification_type, fiftyPercent);
			auto m_joined_T_V = genetic::joinSets(ptrToLoader->getTraningSet(), ptrToLoader->getValidationSet());
			svmEnsemble->scoreLevelWise(m_joined_T_V);

			svmComponents::BaseSvmChromosome ind;
			//ind.updateClassifier(std::make_shared<phd::svm::VotingEnsemble>(repeats[i], weightsPath[i]));
			ind.updateClassifier(svmEnsemble);
			
			svmComponents::BaseSvmChromosome seSvmIndividual;
			seSvmIndividual.updateClassifier(SESVM);
			

			////IN HERE SAVE UNCERTAIN REGIONS FOR OTHER ALGORITHMS
			auto testingPath = std::string(algorithmPath + R"(\Regions)");
			//filesystem::FileSystem fs;
			std::filesystem::create_directories(testingPath);

			auto [certainPathTest, uncertainPathTest] = process_and_save_regions(testingPath, ptrToLoader->getTestSet(), *svmEnsemble, i, "test");

			//phd::data::TabularDataProvider dataProvider;		
			auto certainTest = phd::data::readCsv(certainPathTest);
			auto uncertainTest = phd::data::readCsv(uncertainPathTest);

			//test val train sv length expected sv
			handleCertain(output, certainTest, ind);
			//handleCertainAndUncertainSeSVM(output, certainTest, ind, seSvmIndividual, uncertainTest);
			handleUncertain(output, uncertainTest, ind);
			
			auto [certainPath, uncertainPath] = process_and_save_regions(testingPath, ptrToLoader->getValidationSet(), *svmEnsemble, i, "validation");

			auto certainVal = phd::data::readCsv(certainPath);
			auto uncertainVal = phd::data::readCsv(uncertainPath);

			//test val train sv length expected sv
			handleCertain(output, certainVal, ind);
			//handleCertainAndUncertainSeSVM(output, certainVal, ind, seSvmIndividual, uncertainVal);
			handleUncertain(output, uncertainVal, ind);
			
			auto [certainPathTr, uncertainPathTr] = process_and_save_regions(testingPath, ptrToLoader->getTraningSet(), *svmEnsemble, i, "train");

			auto certainTr = phd::data::readCsv(certainPathTr);
			auto uncertainTr = phd::data::readCsv(uncertainPathTr);

			//test val train sv length expected sv
			handleCertain(output, certainTr, ind);
			//handleCertainAndUncertainSeSVM(output, certainTr, ind, seSvmIndividual, uncertainTr);
			handleUncertain(output, uncertainTr, ind);
			
			output << svmEnsemble->getNumberOfSupportVectors() << "\t";
			output << svmEnsemble->m_classifieres.size() << "\t";
			output << 0 << "\n";
			
		}
		output.close();
	}

	void resulstForSingleRunRegionsGecco(std::filesystem::path fold, std::filesystem::path algorithm, std::string algorithmPath, std::vector<std::string> /*allCSVs*/)
	{
		auto algorithmName = platform::stringUtils::splitString(algorithm.stem().string(), "__")[0];


		if (algorithmName.find("EnsembleList_With_Alma") == std::string::npos)
			return;

		
		auto svmPath = getAllSvms(algorithmPath);

		if (svmPath.empty())
		{
			LOG_F(ERROR, "No svm files inside folder skipping: %s", algorithmPath);
			return;
		}
		
		auto configFile = getJSON(algorithmPath);

		platform::Subtree config(configFile);

		const auto con = genetic::SvmWokrflowConfiguration(config);
		std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
		auto normalize = true;
		auto resample = false;
		ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath, normalize, resample);

		
	

		auto pattern = std::regex("(\\d+)fold");
		//std::map<int, std::vector<std::filesystem::path>> repeats;
		
	/*	for (auto p : svmPath)
		{
			std::smatch sm;
			const auto s = p.string();
			std::regex_search(s, sm, pattern);
			repeats[std::stoi(sm.str().substr(0, sm.str().size() - 4))].emplace_back(p);
		}*/


		std::cout << algorithmPath << "\n";
		std::cout << algorithmName << "\n";

		algorithmName = "Gecco2022_Certain";
		//phd::svm::VotingEnsemble svmEnsemble(svmPath, weightsPath);
		std::ofstream output(fold.string() + "\\" + algorithmName + "_LastNode.txt");

		for (auto i = 0; i < svmPath.size(); ++i)
		{
			auto weigthMock = "file.txt";

			//std::shared_ptr<phd::svm::VotingEnsemble> svmEnsemble  = std::make_shared<phd::svm::VotingEnsemble>(repeats[i], weightsPath[i]); //phd::svm::VotingEnsemble svmEnsemble(repeats[i], weightsPath[i]);

			std::shared_ptr<phd::svm::EnsembleListSvm> svmEnsemble = std::make_shared<phd::svm::EnsembleListSvm>(svmPath[i], true, true);
			//std::shared_ptr<phd::svm::VotingEnsemble> svmEnsemble = std::make_shared<phd::svm::VotingEnsemble>(repeats[i], weigthMock);
			//auto m_joined_T_V = genetic::joinSets(ptrToLoader->getTraningSet(), ptrToLoader->getValidationSet());
			//svmEnsemble->scoreLevelWise(m_joined_T_V);

			svmComponents::BaseSvmChromosome ind;
			//ind.updateClassifier(std::make_shared<phd::svm::VotingEnsemble>(repeats[i], weightsPath[i]));
			ind.updateClassifier(svmEnsemble);


			////IN HERE SAVE UNCERTAIN REGIONS FOR OTHER ALGORITHMS
			auto testingPath = std::string(algorithmPath + R"(\Regions)");
	
			std::filesystem::create_directories(testingPath);

			auto [certainPathTest, uncertainPathTest] = process_and_save_regions(testingPath, ptrToLoader->getTestSet(), *svmEnsemble, i, "test");

			auto certainTest = phd::data::readCsv(certainPathTest);
			auto uncertainTest = phd::data::readCsv(uncertainPathTest);

			//test val train sv length expected sv
			handleCertain(output, certainTest, ind);
			handleUncertain(output, uncertainTest, ind);

			auto [certainPath, uncertainPath] = process_and_save_regions(testingPath, ptrToLoader->getValidationSet(), *svmEnsemble, i, "validation");

			auto certainVal = phd::data::readCsv(certainPath);
			auto uncertainVal = phd::data::readCsv(uncertainPath);

			//test val train sv length expected sv
			handleCertain(output, certainVal, ind);
			handleUncertain(output, uncertainVal, ind);

			auto [certainPathTr, uncertainPathTr] = process_and_save_regions(testingPath, ptrToLoader->getTraningSet(), *svmEnsemble, i, "train");

			auto certainTr = phd::data::readCsv(certainPathTr);
			auto uncertainTr = phd::data::readCsv(uncertainPathTr);

			//test val train sv length expected sv
			handleCertain(output, certainTr, ind);
			handleUncertain(output, uncertainTr, ind);

			output << svmEnsemble->getNumberOfSupportVectors() << "\t";
			//output << svmEnsemble->.size() << "\t";
			output << 0 << "\n";

		}
		output.close();
	}

	
	void resulstForSingleRun(std::filesystem::path fold, std::filesystem::path algorithm, std::string algorithmPath, std::vector<std::string> /*allCSVs*/)
	{
		auto configFile = getJSON(algorithmPath);

		platform::Subtree config(configFile);

		const auto con = genetic::SvmWokrflowConfiguration(config);
		std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
		auto normalize = true;
		auto resample = false;
		ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath, normalize, resample);
		
		auto algorithmName = platform::stringUtils::splitString(algorithm.stem().string(), "__")[0];

		auto svmPath = getAllSvms(algorithmPath);

		auto pattern = std::regex("(\\d+)fold");
		std::map<int, std::vector<std::filesystem::path>> repeats;
		for(auto p : svmPath)
		{
			std::smatch sm;
			const auto s = p.string();
			std::regex_search(s, sm, pattern);
			repeats[std::stoi(sm.str().substr(0,sm.str().size()-4))].emplace_back(p);
		}
		
		
		auto weightsPath = getWeights(algorithmPath);

		std::cout << algorithmPath << "\n";
		std::cout << algorithmName << "\n";

		//phd::svm::VotingEnsemble svmEnsemble(svmPath, weightsPath);
		//std::ofstream output(fold.string() + "\\" + algorithmName + "_ClassificationSchemeTest.txt");
		std::ofstream output(fold.string() + "\\" + algorithmName + "_SESVM_LAST.txt");

		for( auto i = 0; i < repeats.size(); ++i)
		{ 
		
		
		phd::svm::VotingEnsemble svmEnsemble(repeats[i], weightsPath[i]);
		auto m_joined_T_V = genetic::joinSets(ptrToLoader->getTraningSet(), ptrToLoader->getValidationSet());
		svmEnsemble.scoreLevelWise(m_joined_T_V);


		//IN HERE SAVE UNCERTAIN REGIONS FOR OTHER ALGORITHMS
		auto testingPath = std::string(R"(\Regions)");

		process_and_save_regions(testingPath, ptrToLoader->getValidationSet(), svmEnsemble, i, "validation");
		process_and_save_regions(testingPath, ptrToLoader->getTestSet(), svmEnsemble, i, "test");
		process_and_save_regions(testingPath, ptrToLoader->getTraningSet(), svmEnsemble, i, "train");


		//do visualization
		/*if (true)
		{
			strategies::FileSinkStrategy m_savePngElement;
			svmComponents::SvmVisualization visualization3;
			std::filesystem::path m_pngNameSource;
			int i = 0;
			for (auto cl : svmEnsemble.m_classifieres)
			{
				auto image3 = visualization3.createEnsembleVisualization(*cl, 500, 500, ptrToLoader->getTraningSet(), ptrToLoader->getValidationSet(), ptrToLoader->getTestSet());
				genetic::SvmWokrflowConfiguration config_copy3{ "", "", "", fold.string(), "SCListEnsemble_All__" + std::to_string(i) + "__", "" };
				genetic::setVisualizationFilenameAndFormat(svmComponents::imageFormat::png, m_pngNameSource, config_copy3);
				m_savePngElement.launch(image3, m_pngNameSource);
				i++;
			}

			auto image3 = visualization3.createEnsembleVisualization(svmEnsemble, 500, 500, ptrToLoader->getTraningSet(), ptrToLoader->getValidationSet(), ptrToLoader->getTestSet());
			genetic::SvmWokrflowConfiguration config_copy3{ "", "", "", fold.string(), "SCVotingEnsemble_All__", "" };
			setVisualizationFilenameAndFormat(svmComponents::imageFormat::png, m_pngNameSource, config_copy3);
			m_savePngElement.launch(image3, m_pngNameSource);
		}*/
		

		auto [matrixVal, uncertainNumberVal] = evaluateForDataset(svmEnsemble, ptrToLoader->getValidationSet());
		auto [matrixTest, uncertainNumberTest] = evaluateForDataset(svmEnsemble, ptrToLoader->getTestSet());
		//auto [matrixTrain, uncertainNumberTrain] = evaluateForDataset(svmEnsemble, ptrToLoader->getTraningSet());

		

		output << svmComponents::ConfusionMatrix(matrixVal[0][0], matrixVal[1][1], matrixVal[1][0], matrixVal[0][1]) << "\t"
			<< svmComponents::ConfusionMatrix(matrixTest[0][0], matrixTest[1][1], matrixTest[1][0], matrixTest[0][1]) << "\t"
			<< uncertainNumberVal << "\t" << uncertainNumberTest << "\n";
		/*output << svmComponents::ConfusionMatrix(0,0,0,0) << "\t"
			<< svmComponents::ConfusionMatrix(matrixTest[0][0], matrixTest[1][1], matrixTest[1][0], matrixTest[0][1]) << "\t"
			<< 0 << "\t" << uncertainNumberTest << "\n";*/



		
		

			
		}
		output.close();
	}

	void runAlgorithm()
	{
		//auto input_path = R"(D:\ENSEMBLE_868_Regions_certain_05_thr30x)";
		auto input_path = m_config.outputFolder;

		for (auto& dataset : std::filesystem::directory_iterator(input_path))
		{
			if (std::filesystem::is_directory(dataset))
			{
				for (auto fold : std::filesystem::directory_iterator(dataset))
				{
					if (!std::filesystem::is_directory(fold))
						continue;

					std::shared_ptr<phd::svm::libSvmImplementation> SESVM;
					//load SE-SVM to handle uncertain regions
					for (auto algorithm : std::filesystem::directory_iterator(fold))
					{
						std::string algorithmPath((dataset / algorithm).string());
						std::vector<std::string> allCSVs;
						if (std::filesystem::is_directory(algorithmPath))
						{
							auto algorithmName = platform::stringUtils::splitString(algorithm.path().stem().string(), "__")[0];

							
							if (algorithmName.find("SE-SVM") == std::string::npos)
								continue;

							std::cout << algorithmPath << "\n";
							auto svms = getAllSvms(algorithmPath);

							SESVM = std::make_shared<phd::svm::libSvmImplementation>(svms[0]);

						}
					}


					//folders with algorithms in here
					for (auto algorithm : std::filesystem::directory_iterator(fold))
					{
						std::string algorithmPath((dataset / algorithm).string());
						std::vector<std::string> allCSVs;
						if (std::filesystem::is_directory(algorithmPath))
						{
							//resulstForSingleRun(fold, algorithm, algorithmPath, allCSVs);
							resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "All",false, SESVM);
							//resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "Cascade", false);
							//resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "Cascade", true);
							//resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "Cascade_weight", false);
							//resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "Cascade_weight", true);
							//resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "Cascade_node", false);
							//resulstForSingleRunRegions(fold, algorithm, algorithmPath, allCSVs, "Cascade_node", true);

							//resulstForSingleRunRegionsGecco(fold, algorithm, algorithmPath, allCSVs);
						}
					}
				}
			}
		}
	}

private:
	testApp::configTestApp m_config;
};




class RegionsScoresWithRepeats
{
public:
	RegionsScoresWithRepeats(std::string modelsPath, std::string regionsPath)
	{
		m_modelsPath = modelsPath;
		m_regionsPath = regionsPath;
		//m_config = testApp::parseCommandLineArguments(argc, argv);
	}

	std::shared_ptr<phd::svm::ISvm> loadSvm(const std::string& svmPath, const std::string& algorithmName)
	{
		std::shared_ptr<phd::svm::ISvm> svm;
		
		if (algorithmName.find("EnsembleList_With_Alma") != std::string::npos || algorithmName.find("EnsembleList_With_Alma") != std::string::
			npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
		}
		else if (algorithmName.find("EnsembleList_With_SESVM") != std::string::npos || algorithmName.find("EnsembleList_With_SESVM") != std::string::
			npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
		}
		else if (algorithmName.find("EnsembleList_DistanceScheme") != std::string::npos || algorithmName.find("EnsembleList_DistanceScheme") !=
			std::string::npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, false);
		}
		else if (algorithmName.find("EnsembleList_ALMA_no_inheritance") != std::string::npos || algorithmName.find(
			"EnsembleList_ALMA_no_inheritance") != std::string::npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
		}
		else if (algorithmName.find("Baseline_EnsembleList_RBF") != std::string::npos || algorithmName.find(
			"Baseline_EnsembleList_RBF") != std::string::npos)
		{
			svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, false);
		}
		else if(algorithmName.find("BigSetsEnsemble") != std::string::npos || algorithmName.find(
			"BigSetsEnsemble") != std::string::npos)
		{
			//svm = std::make_shared<phd::svm::VotingEnsemble>(svmPath, false);
		}
		else
		{
			//svm = std::make_shared<phd::svm::EnsembleListSvm>(svmPath, true);
			svm = std::make_shared<phd::svm::libSvmImplementation>(svmPath);
		}

		return svm;
	}

	void createOutput(std::shared_ptr<phd::svm::ISvm> svm,
		std::ofstream& output,
		double avg_sv,
		genetic::LocalFileDatasetLoader& certainLoader,
		genetic::LocalFileDatasetLoader& uncertainLoader,
		svmComponents::BaseSvmChromosome& individual)
	{
		handleCertain(output, certainLoader.getTestSet(), individual);
		handleUncertain(output, uncertainLoader.getTestSet(), individual);
		handleCertain(output, certainLoader.getValidationSet(), individual);
		handleUncertain(output, uncertainLoader.getValidationSet(), individual);
		handleCertain(output, certainLoader.getTraningSet(), individual);
		handleUncertain(output, uncertainLoader.getTraningSet(), individual);

		output << svm->getNumberOfSupportVectors() << "\t";

		auto svm_ensemble = dynamic_cast<phd::svm::EnsembleListSvm*>(svm.get());
		if (svm_ensemble)
		{
			output << svm_ensemble->list_length << "\t";
		}
		else
		{
			output << 0 << "\t";
		}

		output << avg_sv << "\n";
	}

	double getSvNumber(std::shared_ptr<phd::svm::ISvm> svm)
	{
		double avg_sv = 0.0;
		auto svm_ensemble = dynamic_cast<phd::svm::EnsembleListSvm*>(svm.get());
		if (svm_ensemble)
		{
			/*	auto nodesSV = svm_ensemble->getNodesNumberOfSupportVectors();

									avg_sv = test_percent[0] * nodesSV[0];

									for(auto i = 1u; i < trainCertain.size(); ++i)
									{
										avg_sv += (test_percent[i] - test_percent[i - 1]) * nodesSV[i];
									}
									avg_sv += (test_percent[test_percent.size() - 1]) * svm_ensemble->getNumberOfSupportVectors();*/
		}
		else
		{
			avg_sv = svm->getNumberOfSupportVectors();
		}
		return avg_sv;
	}

	void getAllRegions(std::vector<std::string> allCSVs, std::vector<std::string>& trainCertain, std::vector<std::string>& uncertainTrain, std::vector<std::string>& certainVAL, std::vector<std::string>& uncertainVAL, std::vector<std::string>& certainTEST, std::vector<std::string>& uncertainTEST)
	{
		trainCertain = filterOut(allCSVs, "train_certain"); //train_certain_2
		uncertainTrain = filterOut(allCSVs, "train_uncertain");
		certainVAL = filterOut(allCSVs, "validation_certain");
		uncertainVAL = filterOut(allCSVs, "validation_uncertain");
		certainTEST = filterOut(allCSVs, "test_certain");
		uncertainTEST = filterOut(allCSVs, "test_uncertain");

		std::sort(trainCertain.begin(), trainCertain.end(), compareNat);
		std::sort(uncertainTrain.begin(), uncertainTrain.end(), compareNat);
		std::sort(certainVAL.begin(), certainVAL.end(), compareNat);
		std::sort(uncertainVAL.begin(), uncertainVAL.end(), compareNat);
		std::sort(certainTEST.begin(), certainTEST.end(), compareNat);
		std::sort(uncertainTEST.begin(), uncertainTEST.end(), compareNat);
	}

	void resulstForSingleRun(std::filesystem::path fold, std::filesystem::path algorithm, std::string algorithmPath, std::string regionsPath, std::vector<std::string> allCSVs, std::string saveFolder)
	{
		allCSVs = getAllCsvs(regionsPath);

		std::vector<std::string> trainCertain;
		std::vector<std::string> uncertainTrain;
		std::vector<std::string> certainVAL;
		std::vector<std::string> uncertainVAL;
		std::vector<std::string> certainTEST;
		std::vector<std::string> uncertainTEST;
		getAllRegions(allCSVs, trainCertain, uncertainTrain, certainVAL, uncertainVAL, certainTEST, uncertainTEST);

		auto algorithmName = platform::stringUtils::splitString(algorithm.stem().string(), "__")[0];
		auto svmPath = getAllSvms(algorithmPath);

		std::cout << algorithmPath << "\n";
		std::cout << algorithmName << "\n";

		std::shared_ptr<phd::svm::ISvm> svm = loadSvm(svmPath[0].string(), algorithmName);

		auto avg_sv = getSvNumber(svm);

		std::ofstream output(saveFolder + "\\" + algorithmName + "_LastNode.txt"); //wciepac w odpowiedni folder i odpowiednie miejsce
		for (auto i = 0; i < trainCertain.size(); ++i)
		{
			auto normalize = false;
			auto resample = false;
			auto certainLoader = genetic::LocalFileDatasetLoader(trainCertain[i], certainVAL[i], certainTEST[i], normalize, resample);
			auto uncertainLoader = genetic::LocalFileDatasetLoader(uncertainTrain[i], uncertainVAL[i], uncertainTEST[i], normalize,
				resample);

			svmComponents::BaseSvmChromosome individual;
			individual.updateClassifier(svm);
			
			createOutput(svm, output, avg_sv, certainLoader, uncertainLoader, individual);
		}
		output.close();
	}

	void runAlgorithm()
	{
		//auto input_path = R"(D:\ENSEMBLE_631_2D_old_ET_V_ClassBias)";
		auto input_path = m_modelsPath;

		auto regions_dataset = std::filesystem::directory_iterator(m_regionsPath);
		for (auto& dataset : std::filesystem::directory_iterator(input_path))
		{
			while(!std::filesystem::is_directory(*regions_dataset))
			{
				std::cout << *regions_dataset << "\n";
				++regions_dataset;
			}
			
			/*std::cout << dataset << "\n";
			std::cout << *regions_dataset << "\n";*/
			
			if (std::filesystem::is_directory(dataset) && std::filesystem::is_directory(*regions_dataset))
			{
				auto regions_fold = std::filesystem::directory_iterator(*regions_dataset);
				for (auto fold : std::filesystem::directory_iterator(dataset))
				{
					std::cout << fold << "\n";
					std::cout << *regions_fold << "\n";
					
					if (!std::filesystem::is_directory(fold) && !std::filesystem::is_directory(*regions_fold))
					{
						++regions_fold;
						continue;
					}

					
					//HERE find folder with BigEnsembles and select regions folder in here, remember that there can be multiple algorithms to be evaluated
					auto regionsAlgorithmFolder = (*regions_fold).path();
					for (const auto& entry : std::filesystem::directory_iterator(*regions_fold))
					{
						if(std::filesystem::is_directory(entry) && entry.path().string().find("BigSetsEnsemble") != std::string::npos)
						{
							regionsAlgorithmFolder = entry.path() / "Regions";
							break;
						}
					}
					
					std::cout << regionsAlgorithmFolder << "\n";
					
					//folders with algorithms in here
					for (auto algorithm : std::filesystem::directory_iterator(fold))
					{
						std::string algorithmPath((dataset / algorithm).string());
						std::vector<std::string> allCSVs;

						if (std::filesystem::is_directory(algorithmPath) && algorithmPath.find("BigSetsEnsemble") != std::string::npos)
							continue;
						
						if (std::filesystem::is_directory(algorithmPath))
						{
							resulstForSingleRun(fold, algorithm, algorithmPath, regionsAlgorithmFolder.string(), allCSVs, (*regions_fold).path().string());
						}
					}
				}
				++regions_fold;
			}
			++regions_dataset;
		}
	}

private:
	testApp::configTestApp m_config;
	std::string m_regionsPath;
	std::string m_modelsPath;
};

//template<class T>
//std::vector<T>make_vector_from_1d_numpy_array(py::array_t<T>py_array)
//{
//	return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
//}
//
//namespace py = pybind11;
//using namespace py::literals;

int main(int argc, char* argv[])
{
	try
	{
		//loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;
		auto config = testApp::parseCommandLineArguments(argc, argv);

		if (std::filesystem::exists(config.outputFolder))
		{
			std::filesystem::create_directories(config.outputFolder);
		}
		std::string logfile(config.outputFolder + "\\experiment_log.log");
		loguru::add_file(logfile.c_str(), loguru::Append, loguru::Verbosity_MAX);

		//C:\Users\Wojtek\Anaconda3\envs\Deeva
		//char str[] = R"(PYTHONPATH=C:\Users\Wojtek\Anaconda3\Lib;C:\Users\Wojtek\Anaconda3\libs;C:\Users\Wojtek\Anaconda3\Lib\site-packages;C:\Users\Wojtek\Anaconda3\DLLs)";
		//char str[] = R"(PYTHONPATH=C:\Users\Wojtek\Anaconda3\envs\Deeva\Lib;C:\Users\Wojtek\Anaconda3\envs\Deeva\libs;C:\Users\Wojtek\Anaconda3\envs\Deeva\Lib\site-packages;C:\Users\Wojtek\Anaconda3\envs\Deeva\DLLs)";
		//_putenv(str);
		//py::initialize_interpreter();



		//auto rs = RegionsScores(argc, argv);
		//rs.runAlgorithm();
		
		
		/*auto normalize = true;
		auto resample = false;
		std::filesystem::path train{ R"(D:\ENSEMBLE_TEST_BED2 - Copy\2D_shapes\1\train.csv)" };
		std::filesystem::path val{ R"(D:\ENSEMBLE_TEST_BED2 - Copy\2D_shapes\1\validation.csv)" };
		std::filesystem::path test{ R"(D:\ENSEMBLE_TEST_BED2 - Copy\2D_shapes\1\test.csv)" };
		auto loader = genetic::LocalFileDatasetLoader(train, val, test, normalize, resample);
		auto testSet = loader.getTestSet();
		auto samples = testSet.getSamples();
		std::vector<float> responses;

		
		auto wrapper = ExtraTreeWrapper();
		wrapper.train(loader.getTraningSet());

		wrapper.save("D:\\test.pkl");

		auto newWrapper = ExtraTreeWrapper();
		newWrapper.load("D:\\test.pkl");
		
		auto start = std::chrono::high_resolution_clock::now();
		responses = newWrapper.predict(loader.getTestSet());
		auto end = std::chrono::high_resolution_clock::now();

		std::cout << "All at once: " << (end - start).count() / 100000 << " ms\n";

 		std::array<std::array<uint32_t, 2>, 2> matrix = { 0 };

		auto targets = testSet.getLabels();
		for(auto i = 0u; i < responses.size(); ++i)
		{
			++matrix[static_cast<int>(responses[i])][static_cast<int>(targets[i])];
		}
		
		svmComponents::ConfusionMatrix cm(matrix[1][1], matrix[0][0], matrix[1][0], matrix[0][1]);

		std::cout << "MCC: " << cm.MCC() << "\n";
		std::cout << "acc: " << cm.accuracy() << "\n";
		std::cout << "F1: " << cm.F1() << "\n";
		std::cout << "pr: " << cm.precision() << "\n";
		std::cout << "re: " << cm.recall() << "\n";*/





		
		//enhancedTrainingSetAndValidationSetExperiments(argc, argv);

		//newMain(argc, argv);
		/*std::vector<double> C = { 0.01, 0.1, 1.0 ,10, 100, 1000 };
		for(auto c : C)
		{
			manual_setting_rbf_linear(c);
		}*/

		//GridSearchLinearRbfFull(argc, argv);

		//svmComponents::DataNormalization::useDefinedMinMax(0, 500);

		//SETUP PYTHON PATH in PythonPath.h with the proper environment

		//RunAlgorithm(argc, argv, AlgorithmName::AKSVM);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_MAX);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_MAX);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_NO_T);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_ONE);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_FS_NO_T);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_FS_MAX);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_FS_ONE);



		//RunAlgorithm(argc, argv, AlgorithmName::GridSearch);
		//RunAlgorithm(argc, argv, AlgorithmName::GASVM);
		//RunAlgorithm(argc, argv, AlgorithmName::MASVM);
		//RunAlgorithm(argc, argv, AlgorithmName::ALGA);
		//RunAlgorithm(argc, argv, AlgorithmName::ALMA);
		//RunAlgorithm(argc, argv, AlgorithmName::SE_SVM);
		//RunAlgorithm(argc, argv, AlgorithmName::ESVM);
		//RunAlgorithm(argc, argv, AlgorithmName::FSALMA);
		//
		//RunAlgorithm(argc, argv, AlgorithmName::SE_SVM_CORRECTED);

		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_MAX);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_NO_T);
		//RunAlgorithm(argc, argv, AlgorithmName::DA_SVM_CE_ONE);
		 
		//newMain(argc, argv);
		//test_metric();

		//RunAlgorithm(argc, argv, AlgorithmName::ALMA);
		//RunAlgorithm(argc, argv, AlgorithmName::SE_SVM);
		//RunAlgorithm(argc, argv, AlgorithmName::SE_SVM_CORRECTED);
		
		newEnsemble(argc, argv);


		//TestClassificationsSchemes test(argc, argv);
		//test.runAlgorithm();

		//rerun_regions_for_plots_of_nodes();

		//rerun_regions();

		//auto cmdConfig = testApp::parseCommandLineArguments(argc, argv);
		//auto rs = RegionsScoresWithRepeats(std::string(R"(D:\ENSEMBLE_851_BASELINE)"), cmdConfig.outputFolder);
		//rs.runAlgorithm();
		
		//auto rs = RegionsScores(argc, argv);
		//rs.runAlgorithm();
		
		/*RunAlgorithm(argc, argv, AlgorithmName::SE_SVM);
		RunAlgorithm(argc, argv, AlgorithmName::ALMA);
		RunAlgorithm(argc, argv, AlgorithmName::GridSearch);
		RunAlgorithm(argc, argv, AlgorithmName::AKSVM);*/
		
		//regionsScoreExperiment(argc, argv);

		//lastRegionsScoreExperiment(argc, argv);
		
		//svmComponents::DataNormalization::useDefinedMinMax(0, 500);
		//GridSearchRbfSubsetsTest(argc, argv);
		//CustomKernelExperiment(argc, argv);

		//RunAlgorithm(argc, argv, AlgorithmName::GridSearch);

		//std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;
		//auto train = std::filesystem::path(R"(C:\PHD\iris_multiclass_all_in_one\Shuttle\1\train.csv)");
		//auto val = std::filesystem::path(R"(C:\PHD\iris_multiclass_all_in_one\Shuttle\1\train.csv)");
		//auto test = std::filesystem::path(R"(C:\PHD\iris_multiclass_all_in_one\Shuttle\1\test.csv)");
		////train02.csv
		//ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(train, val, test);

		//auto trset = ptrToLoader->getTraningSet();

		//phd::svm::libSvmImplementation a;

		//a.setGamma(100);
		//a.setC(100);
		//a.setKernel(phd::svm::KernelTypes::Rbf);
		//a.train(trset);

		//auto realTest = ptrToLoader->getTestSet();
		//svmComponents::ConfusionMatrixMulticlass matrix(a, realTest);
		//std::cout << "Test Balanced acc: " << svmComponents::BalancedAccuracy(matrix);
		//std::cout << "Test Acc: " << svmComponents::Accuracy(matrix);

		//join_multiclass();
		//RunFSALMA(argc, argv);
		//RunGsNoFeatureSelection(argc, argv);

		//newMain(argc, argv);
		//CustomKernelExperiment(argc, argv); //journal and ICPR
		//
		//createAnswerTargegFiles();
		//
		//ConstRbfTest(argc, argv);
		//
		//
		//
		//
		//	//auto basePath = R"(D:\journal_our_models\Original)";
		//auto basePath = R"(D:\journal_our_models\FSALMA)";
		//auto basePath = R"(D:\journal_our_models\Original_working_models)";
		//auto basePath = R"(D:\journal_our_models\Mixed_kernel_coevolution)";
		//auto basePath = R"(D:\journal_our_models\coevolution\joined)";
		//auto basePath = R"(D:\journal_our_models\journal_datasets_Coevolution_with_PSO_and_others)";

		//auto outputBasePath = R"(D:\rerun_experiment_no_thr_mixed_kernel_coevolution)";
		//
		//
		//

		/*auto basePath = R"(D:\test_3_cpp)";
		auto outputPath = R"(D:\DASVM_CE_FS_NOSM_test_mcc)";
		rerun_models2(basePath, outputPath, "test_mcc", "test", "MCC");*/

		////std::vector<std::string> options_test_val{ "test", "val" };
		//std::vector<std::string> options_test_val{ "test", };
		////std::vector<std::string> options_metrics{ "MCC", "F1", "ACC"};
		//std::vector<std::string> options_metrics{ "F1"};
		//
		//for(auto test_val : options_test_val)
		//{
		//	for (auto metric : options_metrics)
		//	{
		//		auto variant_name = test_val + "__" + metric;
		//		/*auto variant_name = std::string("raw_results");
		//		auto metric = "ACC";
		//		auto test_val = "test";*/

		//		//auto basePath = R"(D:\journal_datasets_FSALMA_rerun_results)";
		//		//auto outputPath = R"(D:\FSALMA_)" + variant_name;
		//		//rerun_models2(basePath, outputPath, variant_name, test_val, metric);

		//		//basePath = R"(D:\journal_our_models\Original_working_models)";
		//		//outputPath = R"(D:\ALMA_AND_ICPR_)" + variant_name;
		//		//rerun_models2(basePath, outputPath, variant_name, test_val, metric);

		//		//basePath = R"(D:\journal_dataset_rerun_different_kernels)";
		//		//outputPath = R"(D:\DASVM_VARIANTS_)" + variant_name;
		//		//rerun_models2(basePath, outputPath, variant_name, test_val, metric);

		//		auto basePath = R"(D:\journal_dataset_DASVM_CE_FS_SINGLE)";
		//		auto outputPath = R"(D:\DASVM_TEST_MARGIN_CE_FS_SINGLE_)" + variant_name;
		//		rerun_models2(basePath, outputPath, variant_name, test_val, metric);

		//		//basePath = R"(D:\journal_datasets_COEVOLUTION_CE_FS_GAMMA_MAX)";
		//		//outputPath = R"(D:\COEVOLUTION_CE_FS_GAMMA_MAX_)" + variant_name;
		//		//rerun_models2(basePath, outputPath, variant_name, test_val, metric);

		//	/*	auto basePath = R"(D:\journal_datasets_COEVOLUTION_KERNELS_rerun_results_continue)";
		//		auto outputPath = R"(D:\COEVOLUTION_NOSMO_)" + variant_name;
		//		rerun_models2(basePath, outputPath, variant_name, test_val, metric);
		//		*/
		//		
		//		basePath = R"(D:\journal_dataset_DASVM_CE_FS_NOSMO)";
		//		outputPath = R"(D:\DASVM__TEST_MARGIN_CE_FS_NOSMO)" + variant_name;
		//		rerun_models2(basePath, outputPath, variant_name, test_val, metric);
		//	}
		//}

		/*auto basePath = R"(D:\journal_datasets_FSALMA_rerun_results)";
		auto outputPath = R"(D:\rerun_experiment_recreation_FSALMA)";
		rerun_models2(basePath, outputPath);*/

		/*	basePath = R"(D:\journal_our_models\journal_datasets_Coevolution_with_PSO_and_others)";
			outputPath = R"(D:\rerun_experiment_journal_recreation_coevolution_with_PSO_and_others)";
			rerun_models2(basePath, outputPath);
			
			basePath = R"(D:\journal_dataset_rerun_different_kernels)";
			outputPath = R"(D:\rerun_experiment_journal_recreation_dasvm_other_kernels)";
			rerun_models2(basePath, outputPath);
	
			basePath = R"(D:\journal_our_models\Original_working_models)";
			outputPath = R"(D:\rerun_experiment_journal_recreation_ALMA_ICPR)";
			rerun_models2(basePath, outputPath);*/

		/*auto basePath  = R"(D:\COEVOLUTION_saving_test)";
		auto outputPath = R"(D:\rerun_experiment_no_thr_coevolution)";
		rerun_models2(basePath, outputPath);*/
	}
	catch (const std::exception& e)
	{
		LOG_F(ERROR, "Error: %s", e.what());
	}
	catch (...)
	{
		LOG_F(ERROR, "Something very unpredicted happened and ended the program");
	}
	//py::finalize_interpreter();
	return 0;
}
