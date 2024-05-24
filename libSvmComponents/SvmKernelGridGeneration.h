
#pragma once

#include <memory>
//#include <opencv2/ml.hpp>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libSvmComponents/SvmKernelRandomGeneration.h"

namespace svmComponents
{
/*
 * @wdudzik This will generate population similar to grid search algorithm. Minimum size of population should be
 *          numberOfKernelParameters^2. This way there will be min and max pair of each parameter in population.
 *          The rest of population is generated randomly from distribution. 
 */
class SvmKernelGridGeneration : public geneticComponents::IPopulationGeneration<SvmKernelChromosome>
{
public:
    explicit SvmKernelGridGeneration(std::uniform_real_distribution<double> parameterDistribution,
                                     phd::svm::KernelTypes kernelType,
                                     std::unique_ptr<my_random::IRandomNumberGenerator> rndEngine,
                                     bool isRegression);

    explicit SvmKernelGridGeneration(std::uniform_real_distribution<double> parameterDistribution,
                                     phd::svm::KernelTypes kernelType,
                                     std::unique_ptr<my_random::IRandomNumberGenerator> rndEngine,
                                     bool isRegression,
                                     const dataset::Dataset<std::vector<float>, float>& trainingData);
   
    geneticComponents::Population<SvmKernelChromosome> createPopulation(uint32_t populationSize) override;

private:
    geneticComponents::Population<SvmKernelChromosome> createGridPopulation(uint32_t populationSize) const;
    double calculateLogStep(uint32_t populationSize) const;

    SvmKernelChromosome getDefaultKenerlParameters();

    const std::uniform_real_distribution<double> m_parameterDistribution;
    SvmKernelRandomGeneration m_randomGeneration;
    phd::svm::KernelTypes m_kernelType;
    uint32_t m_parametersNumber;
    bool m_isRegression;
    const dataset::Dataset<std::vector<float>, float>* m_trainingData;
};
} // namespace svmComponents
