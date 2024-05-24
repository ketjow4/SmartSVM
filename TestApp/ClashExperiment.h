#pragma once

#include "Commons.h"

//------------------------------------------------------- Experiments for CLASH Challange -------------------------------------------------------
inline void consolidateExperimentCLASH(filesystem::Path pathToFold,
                                       uint32_t foldNumber,
                                       std::map<uint32_t, KernelParams>& gridSearchResults,
                                       std::vector<std::string>& filters,
                                       bool withFeatureSelection)
{
	try
	{
		genetic::SvmAlgorithmFactory fac;

		auto configs = testApp::getAllConfigFiles(pathToFold);

		const auto con111 = genetic::SvmWokrflowConfiguration(platform::Subtree(filesystem::Path(configs[0])));

		withFeatureSelection = false;

		for (auto& file : configs)
		{
			auto config = platform::Subtree(filesystem::Path(file));

			if (std::find(filters.begin(), filters.end(), config.getValue<std::string>("Name")) == filters.end())
			{
				continue;
			}
			std::cout << file << "\n";

			testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath"));

			const auto summary(config.getValue<std::string>("Svm.OutputFolderPath") +
				genetic::getTimestamp() + "_" + file.filename().string() + "_summary.txt");
			std::ofstream summaryFile(summary);
			std::vector<std::string> logFileNames;

			const auto con = genetic::SvmWokrflowConfiguration(config);
			std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;

			ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath);

			auto repeatNumber = 1;
			if (config.getValue<std::string>("Name") == "GridSearch" ||
				config.getValue<std::string>("Name") == "GridSearchNoFS")
			{
				repeatNumber = 1;
			}

			for (auto i = 0; i < repeatNumber; i++)
			{
				std::cout << "Repeat:" << i << "\n";

				auto configFileName = file.stem().string();
				const auto logFilename = testApp::getLogFilename(foldNumber, i, configFileName);
				config.putValue<std::string>("Svm.TxtLogFilename", logFilename);
				auto al = fac.createAlgorightm(config, *ptrToLoader);
				const auto resultModel = al->run();
				testApp::saveSvmModel(config, foldNumber, i, *resultModel, configFileName);
				testApp::saveSvmResultsToFile(config, foldNumber, i, *resultModel, configFileName, *ptrToLoader);

				if (config.getValue<std::string>("Name") == "GridSearch" || config.getValue<std::string>("Name") == "GridSearchNoFS")
				{
					saveParametersForFold(gridSearchResults, *resultModel, foldNumber);
				}

				logFileNames.push_back(logFilename);
			}

			testApp::createSummaryFile(config, summaryFile, logFileNames);
		}
	}
	catch (const std::runtime_error& e)
	{
		std::cout << e.what() << "\n";
	}
}

void CLASH_experiment(int argc, char* argv[])
{
	auto config = testApp::parseCommandLineArguments(argc, argv);
	//fold number mapped to kernel paramas
	std::map<uint32_t, KernelParams> gridSearchResults;

	std::vector<std::string> dataFolders = testApp::listDirectories(config.datafolder);
	std::map<std::string, testApp::DatasetInfo> datasetInfos;

	for (auto& folder : dataFolders)
	{
		datasetInfos[folder] = (testApp::getInfoAboutDataset(folder + "\\train.csv"));
	}
	//Alga
	const std::vector<uint32_t> Kvalues = {600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500};
	//const std::vector<uint32_t> Kvalues = { 600 };
	for (auto& folder : dataFolders)
	{
		createAlgaConfigsRegression(folder, 1, Kvalues);
	}

	for (auto& datasetFolder : dataFolders)
	{
		{
			auto foldNumber = 1;
			std::vector<std::string> filters{"Alga"}; //@wdudzik note that alma == alga   

			consolidateExperimentCLASH(datasetFolder, foldNumber, gridSearchResults, filters, true);
		}
	}
}
