
#pragma once

#include <filesystem>
#include "libSvmComponents/SvmVisualization.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libSvmComponents/ISvmMetricsCalculator.h"
#include "libSvmComponents/ISvmTraining.h"
#include "libSvmComponents/SvmValidationStrategy.h"


namespace genetic
{
void setVisualizationFilenameAndFormat(svmComponents::imageFormat format,
                                       std::filesystem::path& pngNameSource,
                                       const SvmWokrflowConfiguration& config,
                                       unsigned int generationNumber);


void setVisualizationFilenameAndFormat(svmComponents::imageFormat format,
    std::filesystem::path& pngNameSource,
    const SvmWokrflowConfiguration& config);


void setVisualizationFilenameAndFormatWithPrefix(svmComponents::imageFormat format,
                                                 std::filesystem::path& pngNameSource,
                                                 const SvmWokrflowConfiguration& config,
                                                 unsigned int generationNumber,
                                                 std::string prefix);

std::filesystem::path generateFilenameWithTimestamp(std::filesystem::path filename, std::string prefix, std::filesystem::path outputFolder = "");

template <class chromosome>
void retrainPopulation(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                       const dataset::Dataset<std::vector<float>, float>& validationSet,
                       geneticComponents::Population<chromosome>& population,
                       svmComponents::ISvmTraining<chromosome>& traning,
                       svmComponents::ISvmMetricsCalculator& metric)
{
    if (!population.empty())
    {
        traning.trainPopulation(population, trainingSet);

        auto validation = svmStrategies::SvmValidationStrategy<chromosome>(metric, false);
        population = validation.launch(population, validationSet);
    }
}
} // namespace genetic