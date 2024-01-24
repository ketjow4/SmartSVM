
#pragma once

#include <random>
#include <memory>
//#include <opencv2/ml.hpp>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmComponents/SvmKernelChromosome.h"

namespace svmComponents
{
class SvmKernelRandomGeneration : public geneticComponents::IPopulationGeneration<SvmKernelChromosome>
{
public:
    explicit SvmKernelRandomGeneration(std::uniform_real_distribution<double> parameterDistribution,
                                       phd::svm::KernelTypes kernelType,
                                       std::unique_ptr<random::IRandomNumberGenerator> rndEngine,
                                       bool isRegression);

    geneticComponents::Population<SvmKernelChromosome> createPopulation(uint32_t populationSize) override;

private:
    std::uniform_real_distribution<double> m_parameterDistribution;
    phd::svm::KernelTypes m_kernelType;
    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    uint32_t m_parametersNumber;
    bool m_isRegression;
};
} // namespace svmComponents