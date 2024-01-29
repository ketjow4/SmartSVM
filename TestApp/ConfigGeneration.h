#pragma once

#include <string>
#include <filesystem>
#include "libPlatform/Subtree.h"
#include "libGeneticSvm/MemeticTrainingSetWorkflow.h"
//#include "libFileSystem/FileSystemDefinitions.h"
#include "ConfigParser.h"

namespace testApp
{
inline std::vector<std::string> splitPath(const std::filesystem::path& path)
{
    std::vector<std::string> elements;
    for (auto it(path.begin()), it_end(path.end()); it != it_end; ++it)
    {
        elements.emplace_back(it->string());
    }
    return elements;
}


class ConfigManager
{
public:
    platform::Subtree loadTemplateConfig(const std::filesystem::path& configPath);

    void setRandomNumberGenerators(platform::Subtree& config);

    void setSeedForRng(platform::Subtree& config, int seed);

    void setMetric(platform::Subtree& config); //todo add metric type here

    void setGridKernelInitialPopulationGeneration(platform::Subtree& config); //todo parametrize

    void addKernelParameters(platform::Subtree& config, double C, double gamma);

    void setupDataset(platform::Subtree& config, std::string datafolder, std::string algorithmName, std::string resultFolder);

    void setK(platform::Subtree& config, uint32_t K);

    void setInitialNumberOfFeatures(platform::Subtree& config, uint32_t features);

    void saveConfigToFileFolds(const platform::Subtree& config, std::string datafolder, std::string algorithmName, uint32_t fold);

    void setupStopCondition(platform::Subtree& config);

    void setMetricRegression(platform::Subtree& config);

    void setNumberOfClasses(platform::Subtree& config, uint64_t numberOfClasses);
};
} // namespace testApp
