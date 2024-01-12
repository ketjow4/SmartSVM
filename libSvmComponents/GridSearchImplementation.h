
#pragma once

#include <gsl/span>
#include <SvmLib/OpenCvSvm.h>
#include "libSvmComponents/SvmConfigStructures.h"
#include "libSvmComponents/SvmVisualization.h"
#include "libFilesystem/FileSystemDefinitions.h"

namespace svmComponents
{
/***
 * @wdudzik
 * This class is deprecated and should not be used. Instead use GridSearchCrossValidation.
 */
class GridSearchImplementation
{
public:
    explicit GridSearchImplementation(const GridSearchConfiguration& config,
                                      const dataset::Dataset<std::vector<float>, float> validationData,
                                      const dataset::Dataset<std::vector<float>, float> datasetSamples,
                                      filesystem::Path outputFile);

    phd::svm::OpenCvSvm calculate();
    void calculateGrids() const;
    gsl::span<std::uint8_t> getImage();
    std::string logSvmParameters() const;

private:
    void setupSvmTerminationCriteria(const SvmAlgorithmConfiguration& config);
    void performGridSearch() const;

    const GridSearchConfiguration& m_config;
    dataset::Dataset<std::vector<float>, float> m_validationData;
    dataset::Dataset<std::vector<float>, float> m_testData;
    cv::Ptr<cv::ml::TrainData> m_trainingData;
    cv::Ptr<cv::ml::SVM> m_svm;

    filesystem::Path m_outputFile;
    std::vector<uchar> m_image;
    SvmVisualization m_visualization;
};
} // namespace svmComponents
