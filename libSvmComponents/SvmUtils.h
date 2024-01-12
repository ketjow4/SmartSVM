
#pragma once

#include <random>
#include <gsl/string_span>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/SvmCustomKernelChromosome.h"

namespace svmComponents {
struct SvmAlgorithmConfiguration;
}

namespace svmComponents { namespace svmUtils
{
    uint32_t getNumberOfKernelParameters(phd::svm::KernelTypes kernelType, bool isRegression);

    void setupSvmTerminationCriteria(phd::svm::ISvm& svm, const SvmAlgorithmConfiguration& config);
    
    void setupSvmParameters(phd::svm::ISvm& svm, const SvmKernelChromosome& chromosome);

    std::uniform_real_distribution<double> getRange(const std::string& variablePath, const platform::Subtree& config);

    std::vector<unsigned> countLabels(unsigned numberOfClasses,
                                      const std::vector<DatasetVector>& dataset);

    std::vector<unsigned int> countLabels(unsigned int numberOfClasses,
                                          const dataset::Dataset<std::vector<float>, float>& dataset);

	std::vector<unsigned> countLabels(unsigned numberOfClasses, const std::vector<svmComponents::Gene>& dataset);

    double variance(const dataset::Dataset<std::vector<float>, float>& dataset);

}} // namespace svmComponents::svmUtils