
#include "libSvmComponents/SvmKernelGridGeneration.h"
#include "libSvmComponents/SvmUtils.h"
#include "libSvmComponents/RbfKernel.h"
#include "libSvmComponents/SvmKernelRandomGeneration.h"
#include "LinearKernel.h"
#include "PolyKernel.h"
#include "libPlatform/loguru.hpp"

namespace svmComponents
{
SvmKernelGridGeneration::SvmKernelGridGeneration(std::uniform_real_distribution<double> parameterDistribution,
                                                 phd::svm::KernelTypes kernelType,
                                                 std::unique_ptr<random::IRandomNumberGenerator> rndEngine,
                                                 bool isRegression)
    : m_parameterDistribution([&parameterDistribution]()
    {
        if(parameterDistribution.min() <= 0.0)
        {
            throw ValueNotInRange("kernelParameterDistribution min",
                                  parameterDistribution.min(),
                                  0.0 + std::numeric_limits<double>::epsilon(),
                                  std::numeric_limits<double>::max());
        }
        return parameterDistribution;
    }())
    , m_randomGeneration(parameterDistribution, kernelType, std::move(rndEngine), isRegression)
    , m_kernelType(kernelType)
    , m_parametersNumber(svmUtils::getNumberOfKernelParameters(m_kernelType, isRegression))
    , m_isRegression(isRegression)
	, m_trainingData(nullptr)
{
}

SvmKernelGridGeneration::SvmKernelGridGeneration(std::uniform_real_distribution<double> parameterDistribution,
                                                 phd::svm::KernelTypes kernelType,
                                                 std::unique_ptr<random::IRandomNumberGenerator> rndEngine,
                                                 bool isRegression,
                                                 const dataset::Dataset<std::vector<float>, float>& trainingData)
    : SvmKernelGridGeneration(parameterDistribution, kernelType, std::move(rndEngine), isRegression)
{
    m_trainingData = &trainingData;
}

geneticComponents::Population<SvmKernelChromosome> SvmKernelGridGeneration::createGridPopulation(uint32_t populationSize) const
{
    const ParamGrid parameterGrid(m_parameterDistribution.min(), m_parameterDistribution.max(), calculateLogStep(populationSize));
    
    switch (m_kernelType)
    {
    case phd::svm::KernelTypes::Rbf:
    {
        RbfKernel kernel{parameterGrid, parameterGrid, m_isRegression};
        return kernel.createGridSearchPopulation();
    }
    case phd::svm::KernelTypes::Linear:
    {
	    LinearKernel kernel{parameterGrid, m_isRegression};
	    return kernel.createGridSearchPopulation();
    }
    case phd::svm::KernelTypes::Poly:
    {
        PolyKernel kernel{ parameterGrid, ParamGrid(2,8, 1), m_isRegression };
        return kernel.createGridSearchPopulation();
    }
    default:
        throw GridSearchUnsupportedKernelTypeException(m_kernelType);
    }
}

double SvmKernelGridGeneration::calculateLogStep(uint32_t populationSize) const
{
    const auto rootIndex = 1.0 / m_parametersNumber;
    const auto maxElementInGridParameter = static_cast<unsigned int>(std::floor(std::pow(populationSize, rootIndex)));

    const auto rootIndexForLogarithmicStep = 1.0 / (maxElementInGridParameter - 1);
    return std::pow(m_parameterDistribution.max() / m_parameterDistribution.min(), rootIndexForLogarithmicStep);
}

SvmKernelChromosome SvmKernelGridGeneration::getDefaultKenerlParameters()
{
    if(m_isRegression)
    {
        switch (m_kernelType)
        {
        case phd::svm::KernelTypes::Rbf:
        {
            SvmKernelChromosome defaultOne(phd::svm::KernelTypes::Rbf, { 1,1, 0.001 }, m_isRegression);
            return defaultOne;
        }
        case phd::svm::KernelTypes::Linear:
        {
            SvmKernelChromosome defaultOne(phd::svm::KernelTypes::Linear, { 1,0.001 }, m_isRegression);
            return defaultOne;
        }
        case phd::svm::KernelTypes::Poly:
        {
            //c=1, degree=3, coef0 = 0
            SvmKernelChromosome defaultOne(phd::svm::KernelTypes::Poly, { 1,3,0, 0.001 }, m_isRegression);
            return defaultOne;
        }
        default:
            throw GridSearchUnsupportedKernelTypeException(m_kernelType);
        }
    }
	
    switch (m_kernelType)
    {
    case phd::svm::KernelTypes::Rbf:
    {

        if(m_trainingData)
        {
            auto variance_value = svmComponents::svmUtils::variance(*m_trainingData);
            double gamma_value = 0.0;
            if (variance_value != 0.0)
            {
                gamma_value = 1.0 / (m_trainingData->getSample(0).size() * variance_value);
                
            }
            else
            {
            	gamma_value = 1.0 / static_cast<double>(m_trainingData->getSample(0).size());
            }
            LOG_F(INFO, "Added default sci-kit params to population: C=1, gamma=%f", gamma_value);
        }
        else
        {
	        SvmKernelChromosome defaultOne(phd::svm::KernelTypes::Rbf, { 1,1 }, m_isRegression);
	        return defaultOne;
        }
    }
    case phd::svm::KernelTypes::Linear:
    {
        SvmKernelChromosome defaultOne(phd::svm::KernelTypes::Linear, { 1 }, m_isRegression);
        return defaultOne;
    }
    case phd::svm::KernelTypes::Poly:
    {
    	//c=1, degree=3, coef0 = 0
        SvmKernelChromosome defaultOne(phd::svm::KernelTypes::Poly, { 1,3,0 }, m_isRegression);
        return defaultOne;
    }
    default:
        throw GridSearchUnsupportedKernelTypeException(m_kernelType);
    }
}

geneticComponents::Population<SvmKernelChromosome> SvmKernelGridGeneration::createPopulation(uint32_t populationSize)
{
    if (populationSize < std::pow(m_parametersNumber, 2))
    {
        throw TooSmallPopulationSize(populationSize, static_cast<unsigned int>(std::pow(m_parametersNumber, 2)));
    }

    std::vector<SvmKernelChromosome> individuals;
    individuals.reserve(populationSize);

    auto gridPopulation = createGridPopulation(populationSize);
    individuals.insert(individuals.end(), gridPopulation.begin(), gridPopulation.end());
    auto supplementSize = static_cast<unsigned int>(populationSize - gridPopulation.size());;

    if (gridPopulation.size() > populationSize)
    {
        supplementSize = 0;
        individuals.resize(populationSize);
    }
	
    if (supplementSize > 0)
    {
        if(m_isRegression)
        {
            auto defaultOne = getDefaultKenerlParameters();
            individuals.emplace_back(defaultOne);
        }
        else
        {
            auto defaultOne = getDefaultKenerlParameters();
            individuals.emplace_back(defaultOne);
        }
        if(supplementSize - 1 > 0)
        {
	        auto restOfPopulation = m_randomGeneration.createPopulation(supplementSize - 1);
	        individuals.insert(individuals.end(), restOfPopulation.begin(), restOfPopulation.end());
        }
		
    }

    return geneticComponents::Population<SvmKernelChromosome>(individuals);
}
} // namespace svmComponents
