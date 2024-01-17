
#pragma once

#include <filesystem>
#include "libPlatform/Subtree.h"
//#include "AppUtils/Verbosity.h"

namespace genetic
{
struct SvmWokrflowConfiguration
{
    explicit SvmWokrflowConfiguration(const platform::Subtree& config);

    SvmWokrflowConfiguration(std::filesystem::path trainingDataPath,
                             std::filesystem::path validationDataPath,
                             std::filesystem::path testDataPath,
                             std::filesystem::path outputFolderPath,
                             std::string visualizationFilename,
                             std::string txtLogFilename
							 //testApp::Verbosity verbosity = testApp::Verbosity::Standard
                             );

    const std::filesystem::path trainingDataPath;
    const std::filesystem::path validationDataPath;
    const std::filesystem::path testDataPath;
    std::filesystem::path outputFolderPath;
    const std::string visualizationFilename;
    const std::string txtLogFilename;
    //const testApp::Verbosity verbosity;
};
} // namespace genetic