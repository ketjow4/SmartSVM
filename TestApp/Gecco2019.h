#pragma once

#include <algorithm>
#include <string>
#include "ConfigParser.h"
#include "AppUtils/AppUtils.h"
#include "ConfigGeneration.h"

#include "libGeneticSvm/DefaultWorkflowConfigs.h"
#include "AppUtils/PythonFeatureSelection.h"
#include "Commons.h"
#include "libPlatform/StringUtils.h"

//------------------------------------------------------- Experiments for Gecoo 2019 -------------------------------------------------------
inline void consolidateExperiment(std::filesystem::path pathToFold,
                                  uint32_t foldNumber,
                                  std::map<uint32_t, KernelParams>& gridSearchResults,
                                  std::vector<std::string>& filters,
                                  bool usuMutualInformationOrRFE)
{
	try
	{
		genetic::SvmAlgorithmFactory fac;

		auto configs = testApp::getAllConfigFiles(pathToFold);

		const auto con111 = genetic::SvmWokrflowConfiguration(platform::Subtree(std::filesystem::path(configs[0])));

		if (!usuMutualInformationOrRFE)
		{
			runMutualInfo(con111.trainingDataPath);
		}

		std::vector<bool> featureMask;
		if (usuMutualInformationOrRFE)
		{
			featureMask = runFeatureSelection(con111.trainingDataPath);
		}

		for (auto& file : configs)
		{
			auto config = platform::Subtree(std::filesystem::path(file));

			if (std::find(filters.begin(), filters.end(), config.getValue<std::string>("Name")) == filters.end())
			{
				continue;
			}
			std::cout << file << "\n";

			testApp::createOutputFolder(config.getValue<std::string>("Svm.OutputFolderPath"));

			const auto summary(config.getValue<std::string>("Svm.OutputFolderPath") +
				timeUtils::getTimestamp() + "_" + file.filename().string() + "_summary.txt");
			std::ofstream summaryFile(summary);
			std::vector<std::string> logFileNames;

			const auto con = genetic::SvmWokrflowConfiguration(config);
			std::unique_ptr<genetic::LocalFileDatasetLoader> ptrToLoader;

			if (usuMutualInformationOrRFE)
			{
				ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath, featureMask);
			}
			else
			{
				ptrToLoader = std::make_unique<genetic::LocalFileDatasetLoader>(con.trainingDataPath, con.validationDataPath, con.testDataPath);
			}

			auto repeatNumber = 10;
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

			testApp::createSummaryFile(config, summaryFile, logFileNames, testApp::Verbosity::Standard);
		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}
