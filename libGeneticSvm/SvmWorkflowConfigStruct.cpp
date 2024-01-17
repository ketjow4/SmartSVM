#include "SvmWorkflowConfigStruct.h"

namespace genetic
{
SvmWokrflowConfiguration::SvmWokrflowConfiguration(const platform::Subtree& config)
	: SvmWokrflowConfiguration(config.getValue<std::filesystem::path>("Svm.TrainingData"),
	                           config.getValue<std::filesystem::path>("Svm.ValidationData"),
	                           config.getValue<std::filesystem::path>("Svm.TestData"),
	                           config.getValue<std::filesystem::path>("Svm.OutputFolderPath"),
	                           config.getValue<std::string>("Svm.Visualization.Filename"),
	                           config.getValue<std::string>("Svm.TxtLogFilename")
	                           //testApp::verbosityFromString(config.getValue<std::string>("Svm.LogVerbosity"))
							   )
{
}

SvmWokrflowConfiguration::SvmWokrflowConfiguration(std::filesystem::path trainingDataPath,
                                                   std::filesystem::path validationDataPath,
                                                   std::filesystem::path testDataPath,
                                                   std::filesystem::path outputFolderPath,
                                                   std::string visualizationFilename,
                                                   std::string txtLogFilename
                                                   //testApp::Verbosity verbosity_
												   )
	: trainingDataPath(std::move(trainingDataPath))
	, validationDataPath(std::move(validationDataPath))
	, testDataPath(std::move(testDataPath))
	, outputFolderPath(std::move(outputFolderPath))
	, visualizationFilename(std::move(visualizationFilename))
	, txtLogFilename(std::move(txtLogFilename))
	//, verbosity(verbosity_)
{
}
} // namespace genetic
