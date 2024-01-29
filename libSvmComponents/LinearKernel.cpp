#include "libSvmComponents/SvmComponentsExceptions.h"
#include "LinearKernel.h"

namespace svmComponents
{
LinearKernel::LinearKernel(ParamGrid cGrid, bool isRegression)
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
	, m_isRegression(isRegression)
{
}

void LinearKernel::calculateGrids(const BaseSvmChromosome& individual)
{
	m_cGrid = calculateNewGrid(m_cGrid, individual.getClassifier()->getC());
}

geneticComponents::Population<SvmKernelChromosome> LinearKernel::createGridSearchPopulation()
{
	std::vector<SvmKernelChromosome> population;

	for (auto i = 0u; i <= calculateNumberOfSteps(m_cGrid); ++i)
	{
		if (m_isRegression)
		{
			population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::Linear,
			                                            std::vector<double>{
				                                            m_cGrid.minVal * std::pow(m_cGrid.logStep, i),
				                                            0.001 //epsilon value
			                                            }, m_isRegression));
		}
		else
		{
			population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::Linear,
			                                            std::vector<double>{
				                                            m_cGrid.minVal * std::pow(m_cGrid.logStep, i),
			                                            }, m_isRegression));
		}
	}
	return geneticComponents::Population<SvmKernelChromosome>(population);
}

// void LinearKernel::calculateGrids()
// {
// 	m_cGrid = calculateNewGrid(m_cGrid, m_svm->getC());
// }

// std::string LinearKernel::logSvmParameters()
// {
// 	return "Svm parameters(C):\t" +
// 			std::to_string(m_svm->getC());
// }

// void LinearKernel::performGridSearch(cv::Ptr<cv::ml::TrainData> trainingSet, unsigned numberOfFolds)
// {
// 	m_svm->trainAuto(trainingSet, numberOfFolds, m_cGrid);
// }
} // namespace svmComponents
