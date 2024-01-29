
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "RbfKernel.h"

namespace svmComponents
{
RbfKernel::RbfKernel(ParamGrid cGrid, ParamGrid gammaGrid, bool isRegression)
    : m_cGrid([&cGrid]()
    {
        if (cGrid.minVal <= 0.0)
        {
            throw ValueNotInRange("cGrid.minVal",
                                  cGrid.minVal,
                                  0.0 + std::numeric_limits<double>::epsilon(),
                                  std::numeric_limits<double>::max());
        }
        return cGrid;
    }())
    , m_gammaGrid([&gammaGrid]()
    {
        if (gammaGrid.minVal <= 0.0)
        {
            throw ValueNotInRange("gammaGrid.minVal",
                                  gammaGrid.minVal,
                                  0.0 + std::numeric_limits<double>::epsilon(),
                                  std::numeric_limits<double>::max());
        }
        return gammaGrid;
    }())
    , m_isRegression(isRegression)
{
}

void RbfKernel::calculateGrids(const BaseSvmChromosome& individual)
{
    m_cGrid = calculateNewGrid(m_cGrid, individual.getClassifier()->getC());
    m_gammaGrid = calculateNewGrid(m_gammaGrid, individual.getClassifier()->getGamma());
}

geneticComponents::Population<SvmKernelChromosome> RbfKernel::createGridSearchPopulation()
{
    std::vector<SvmKernelChromosome> population;

    for (auto i = 0u; i <= calculateNumberOfSteps(m_cGrid); ++i)
    {
        for (auto j = 0u; j <= calculateNumberOfSteps(m_gammaGrid); ++j)
        {
            if (m_isRegression)
            {
                population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::Rbf,
                                                            std::vector<double>{
                                                                m_cGrid.minVal * std::pow(m_cGrid.logStep, i),
                                                                m_gammaGrid.minVal * std::pow(m_gammaGrid.logStep, j),
                                                                0.001   //epsilon value
                                                            }, m_isRegression));
            }
            else
            {
                population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::Rbf,
                                                            std::vector<double>{
                                                                m_cGrid.minVal * std::pow(m_cGrid.logStep, i),
                                                                m_gammaGrid.minVal * std::pow(m_gammaGrid.logStep, j)
                                                            }, m_isRegression));
            }
        }
    }
    return geneticComponents::Population<SvmKernelChromosome>(population);
}


// void RbfKernel::calculateGrids()
// {
//     m_cGrid = calculateNewGrid(m_cGrid, m_svm->getC());
//     m_gammaGrid = calculateNewGrid(m_gammaGrid, m_svm->getGamma());
// }

// std::string RbfKernel::logSvmParameters()
// {
//     return "Svm parameters(gamma, C):\t" +
//         std::to_string(m_svm->getGamma()) + "\t" +
//         std::to_string(m_svm->getC());
// }

// void RbfKernel::performGridSearch(cv::Ptr<TrainData> trainingSet, unsigned numberOfFolds)
// {
//     m_svm->trainAuto(trainingSet, numberOfFolds, m_cGrid, m_gammaGrid);
// }
} // namespace svmComponents
