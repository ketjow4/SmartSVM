
#include <chrono>
#include "GridSearchImplementation.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"
#include "SvmAucMetric.h"

namespace svmComponents
{
GridSearchImplementation::GridSearchImplementation(const GridSearchConfiguration& config,
                                                   const dataset::Dataset<std::vector<float>, float> validationData,
                                                   const dataset::Dataset<std::vector<float>, float> datasetSamples,
                                                   filesystem::Path outputFile)
    : m_config(config)
    , m_validationData(validationData)
    , m_testData(datasetSamples)
    , m_outputFile(outputFile)
{
    if (m_validationData.empty())
    {
        throw EmptyDatasetException(DatasetType::Validation);
    }
    if (m_testData.empty())
    {
        throw EmptyDatasetException(DatasetType::Test);
    }

    auto svmTemp = phd::svm::OpenCvSvm();
    svmTemp.setKernel(config.m_svmConfig.m_kernelType);
    svmTemp.setType(phd::svm::SvmTypes::CSvc);

    m_trainingData = svmTemp.createTrainingData(m_validationData);

    m_svm = cv::ml::SVM::create();
    m_svm->setKernel(static_cast<int>(config.m_svmConfig.m_kernelType));
    m_svm->setType(cv::ml::SVM::C_SVC);
    setupSvmTerminationCriteria(config.m_svmConfig);

    m_config.m_kernel->setSvm(m_svm);
}

void GridSearchImplementation::performGridSearch() const
{
    m_config.m_kernel->performGridSearch(m_trainingData, m_config.m_numberOfFolds);
}

void GridSearchImplementation::calculateGrids() const
{
    m_config.m_kernel->calculateGrids();
}

gsl::span<std::uint8_t> GridSearchImplementation::getImage()
{
    if (m_image.empty())
    {
        return gsl::span<std::uint8_t>();
    }

    return gsl::make_span(m_image);
}

std::string GridSearchImplementation::logSvmParameters() const
{
    return m_config.m_kernel->logSvmParameters();
}

phd::svm::OpenCvSvm GridSearchImplementation::calculate()
{
    using clock = std::chrono::high_resolution_clock;
    using milis = std::chrono::duration<double, std::milli>;
    auto start = clock::now();

    performGridSearch();
    auto svmWrapped = std::make_unique<phd::svm::OpenCvSvm>(m_svm);

    if (dynamic_cast<SvmAucMetric*>(&*m_config.m_svmConfig.m_estimationMethod) != nullptr)
    {
        svmWrapped->calculateSigmoidParametrs(m_validationData);
    }

    BaseSvmChromosome svmContainer;
    svmContainer.updateClassifier(std::move(svmWrapped));
    auto validationTimeStart = clock::now();
    const auto validationFitness = m_config.m_svmConfig.m_estimationMethod->calculateMetric(svmContainer, m_validationData);
    const auto validationTime = std::chrono::duration_cast<milis>(clock::now() - validationTimeStart).count();

    auto testTimeStart = clock::now();
    const auto testFitness = m_config.m_svmConfig.m_estimationMethod->calculateMetric(svmContainer, m_testData);
    const auto testTime = std::chrono::duration_cast<milis>(clock::now() - testTimeStart).count();

    const auto end = clock::now();
    const auto duration = std::chrono::duration_cast<milis>(end - start).count();

    constexpr auto maxPredictedLogMessageLength = 100;
    std::string logInfo;
    logInfo.reserve(maxPredictedLogMessageLength);
    logInfo += std::to_string(duration).append("\t");
    logInfo += std::to_string(validationFitness.m_fitness).append("\t");
    logInfo += std::to_string(m_svm->getSupportVectors().rows).append("\t");
    logInfo += std::to_string(testFitness.m_fitness).append("\t");
    logInfo += logSvmParameters().append("\t");
    logInfo += std::to_string(validationTime).append("\t");
    logInfo += std::to_string(testTime);

    std::ofstream outputFile(m_outputFile, std::ios_base::out | std::ios_base::app);
    if (validationFitness.m_confusionMatrix)
    {
        outputFile << logInfo << validationFitness.m_confusionMatrix << "\n";
    }
    else
    {
        outputFile << logInfo << "\n";
    }
    outputFile.close();

    if (m_config.m_svmConfig.m_doVisualization)
    {
        m_image = m_visualization.createVisualization(*svmContainer.getClassifier(),
                                                      m_config.m_svmConfig.m_height,
                                                      m_config.m_svmConfig.m_width,
                                                      m_validationData, m_testData);
    }
    return phd::svm::OpenCvSvm(m_svm);
}

void GridSearchImplementation::setupSvmTerminationCriteria(const SvmAlgorithmConfiguration& config)
{
    if (config.m_useSvmIteration)
    {
        m_svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                                config.m_svmIterationNumber,
                                                config.m_svmEpsilon));
    }
    else
    {
        constexpr int numberOfIterations = 0;
        m_svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS, numberOfIterations, config.m_svmEpsilon));
    }
}
} // namespace svmComponents
