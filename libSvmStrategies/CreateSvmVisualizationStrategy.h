#pragma once

#include <gsl/gsl>
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "libSvmComponents/SvmConfigStructures.h"
#include "libSvmComponents/SvmVisualization.h"
#include "LibGeneticComponents/Population.h"


namespace svmStrategies
{
template <typename chromosome>
class CreateSvmVisualizationStrategy
{
public:
    explicit CreateSvmVisualizationStrategy(const svmComponents::SvmAlgorithmConfiguration& config);

    std::string getDescription() const;
    gsl::span<std::uint8_t> launch(const geneticComponents::Population<chromosome>& population,
                                   const dataset::Dataset<std::vector<float>, float>& trainingData,
                                   const dataset::Dataset<std::vector<float>, float>& validationData);

private:
    const svmComponents::SvmAlgorithmConfiguration& m_config;
    std::vector<unsigned char> m_image;
};

template <typename chromosome>
CreateSvmVisualizationStrategy<chromosome>::CreateSvmVisualizationStrategy(const svmComponents::SvmAlgorithmConfiguration& config)
    : m_config(config)
{
}

template <typename chromosome>
std::string CreateSvmVisualizationStrategy<chromosome>::getDescription() const
{
    return "Create visualization for 2D SVM classifier";
}

template <typename chromosome>
gsl::span<std::uint8_t> CreateSvmVisualizationStrategy<chromosome>::launch(const geneticComponents::Population<chromosome>& population,
                                                                           const dataset::Dataset<std::vector<float>, float>& trainingData,
                                                                           const dataset::Dataset<std::vector<float>, float>& validationData)
{
    try
    {
        auto svm = population.getBestOne().getClassifier();
        svmComponents::SvmVisualization visualization;
        m_image = visualization.createVisualization(*svm, m_config.m_height, m_config.m_width, trainingData, validationData);

        return gsl::make_span(m_image);
    }
    catch (const svmComponents::UntrainedSvmClassifierException& /*exception*/)
    {
        throw;
        //handleException(exception);
    }
    catch (...)
    {
        throw;
        //m_logger.LOG(logger::LogLevel::Error, "Unknown error in CreatePopulationStrategy");
    }
}
} // namespace svmStrategies
