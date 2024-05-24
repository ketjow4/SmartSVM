
#include "libGeneticComponents/GeneticExceptions.h"
#include "SvmKernelRandomGeneration.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"

namespace svmComponents
{
SvmKernelRandomGeneration::SvmKernelRandomGeneration(std::uniform_real_distribution<double> parameterDistribution,
                                                     phd::svm::KernelTypes kernelType,
                                                     std::unique_ptr<my_random::IRandomNumberGenerator> rndEngine,
                                                     bool isRegression)
    : m_parameterDistribution(std::move(parameterDistribution))
    , m_kernelType(kernelType)
    , m_rngEngine(std::move(rndEngine))
    , m_parametersNumber(svmUtils::getNumberOfKernelParameters(m_kernelType, isRegression))
    , m_isRegression(isRegression)
{
}

geneticComponents::Population<SvmKernelChromosome> SvmKernelRandomGeneration::createPopulation(uint32_t populationSize)
{
    if (populationSize == 0)
    {
        constexpr auto minumumPopulationSize = 1u;
        throw TooSmallPopulationSize(populationSize, minumumPopulationSize);
    }

    std::vector<SvmKernelChromosome> population(populationSize);

    std::generate(population.begin(), population.end(), [this]
                  {
                      std::vector<double> parameters(m_parametersNumber);
                      std::generate(parameters.begin(), parameters.end(), [this]
                                    {
                                        return m_rngEngine->getRandom(m_parameterDistribution);
                                    });
                      return SvmKernelChromosome(m_kernelType, parameters, m_isRegression);
                  });

    return geneticComponents::Population<SvmKernelChromosome>(population);
}
} // namespace svmComponents