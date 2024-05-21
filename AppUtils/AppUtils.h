#pragma once

#include <string>

#include "libPlatform/Subtree.h"

#include "libGeneticSvm/SvmAlgorithmFactory.h"
#include "libGeneticSvm/FeatureSelectionWorkflow.h"
#include "libGeneticSvm/MemeticTrainingSetWorkflow.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/LocalFileDatasetLoader.h"

#include "libPlatform/Verbosity.h"

namespace testApp
{
	std::string getLastLine(std::string folder, std::string filename);

	void createSummaryFile(platform::Subtree& config, std::ofstream& summaryFile, std::vector<std::string>& logFileNames, platform::Verbosity verbosity);
	std::string createOutputFolder(std::string& outputFolderName);
	std::string createOutputFolderWithDetails(std::string& outputFolderName, std::string& expDetails);
	std::string getLogFilename(int fold, int i, const std::string& configFile);

	void dumpVectorToFile(filesystem::DiskFile& resultFile, const std::vector<float>& minVec);
	void loadDataAndDumpScalingValues(genetic::SvmWokrflowConfiguration& con, genetic::LocalFileDatasetLoader& dataLoading);

	void saveSvmModel(platform::Subtree& config, int fold, int i, phd::svm::ISvm& resultModel, const std::string& configFile);

	void saveSvmResultsToFile(platform::Subtree& config, int fold, int i, phd::svm::ISvm& resultModel, const std::string& configFile,
		genetic::IDatasetLoader& loadingWorkflow);

	void saveSvmResultsToFile2(std::string outputfolder, int i, phd::svm::ISvm& resultModel,
							   genetic::IDatasetLoader& loadingWorkflow, bool useFeateures = false);

	void saveSvmResultsToFile3(std::string outputfolder, int i, phd::svm::ISvm& resultModel,
		genetic::IDatasetLoader& loadingWorkflow);

	void saveSvmGroupsResultsToFile(platform::Subtree& config, int fold, int i, phd::svm::ISvm& resultModel, const std::string& configFile,
		genetic::IDatasetLoader& loadingWorkflow);

	void saveRepositoryState(const std::filesystem::path& outPath);

	struct configTestApp
	{
		std::filesystem::path configFile;
		std::string datafolder;
		std::string outputFolder;
		platform::Verbosity verbosity;
	};

	configTestApp parseCommandLineArguments(int argc, char** argv);

	std::vector<std::filesystem::path> getAllConfigFiles(std::filesystem::path folderPath);

	std::vector<std::string> listDirectories(const std::filesystem::path& path);

	struct DatasetInfo
	{
		DatasetInfo(uint64_t size = 0, uint64_t numberOfFeatures = 0, std::vector<uint32_t> values = {}, uint64_t numberOfClasses = 2)
			: size(size)
			, numberOfFeatures(numberOfFeatures)
			, kValues(std::move(values))
			, numberOfClasses(numberOfClasses)
		{
		}

		uint64_t size;
		uint64_t numberOfFeatures;
		std::vector<uint32_t> kValues;
		uint64_t numberOfClasses; //number of unique labels in dataset
	};

	std::vector<unsigned int> countLabels2(unsigned int numberOfClasses,
		const dataset::Dataset<std::vector<float>, float>& dataset);

	DatasetInfo getInfoAboutDataset(const std::filesystem::path& path);
} // namespace platform
