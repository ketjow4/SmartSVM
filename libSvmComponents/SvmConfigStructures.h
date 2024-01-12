
#pragma once

#include <memory>
#include <unordered_map>
#include <opencv2/ml.hpp>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/BaseKernelGridSearch.h"
#include "libSvmComponents/ISvmMetricsCalculator.h"
#include "libSvmComponents/SvmVisualization.h"
#include "SvmMetricFactory.h"
#include "ISvmTraining.h"
#include "SvmLib/SvmFactory.h"

namespace svmComponents
{
struct SvmAlgorithmConfiguration
{
    explicit SvmAlgorithmConfiguration(const platform::Subtree& config);

    const phd::svm::KernelTypes m_kernelType;
    const double m_svmEpsilon;
    const bool m_useSvmIteration;
    const unsigned int m_svmIterationNumber;
    const svmMetricType m_estimationType;
    const std::shared_ptr<ISvmMetricsCalculator> m_estimationMethod;
    const std::shared_ptr<IGroupPropagation> m_groupPropagationMethod;
    const bool m_doVisualization;
    const unsigned int m_height;
    const unsigned int m_width;
    const imageFormat m_visualizationFormat;
    phd::svm::SvmImplementationType m_implementationType;

private:
    const static std::unordered_map<std::string, svmMetricType> m_translations;
};

struct GridSearchConfiguration
{
    explicit GridSearchConfiguration(const platform::Subtree& config);

	explicit GridSearchConfiguration(SvmAlgorithmConfiguration svmConfig,
									 unsigned int numberOfFolds,
									 unsigned int numberOfIterations,
									 unsigned int subsetSize,
									 unsigned int subsetIterations,
									 std::shared_ptr<ISvmTraining<SvmKernelChromosome>> training,
									 std::shared_ptr<BaseKernelGridSearch> kernel);

    static cv::ml::ParamGrid parseGridParameters(const std::string& gridName, const platform::Subtree& config);

    const SvmAlgorithmConfiguration m_svmConfig;
    const unsigned int m_numberOfFolds;
    const unsigned int m_numberOfIterations;
	const unsigned int m_subsetSize;
	const unsigned int m_subsetIterations;

    const std::shared_ptr<ISvmTraining<SvmKernelChromosome>> m_training;
    const std::shared_ptr<BaseKernelGridSearch> m_kernel;

private:
    std::shared_ptr<BaseKernelGridSearch> createKernelGrid(const phd::svm::KernelTypes kernelType,
                                                           const platform::Subtree& config) const;

    static void validateGridName(const std::string& gridName);
};
} // namespace svmComponents
