#include "libSvmComponents/SvmComponentsExceptions.h"
#include "PolyKernel.h"

namespace svmComponents
{
PolyKernel::PolyKernel(ParamGrid cGrid, ParamGrid degreeGrid, bool isRegression)
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
	, m_degreeGrid([&degreeGrid]()
		{
			if (degreeGrid.minVal <= 0.0)
			{
				throw ValueNotInRange("degreeGrid.minVal",
				                      degreeGrid.minVal,
				                      0.0 + std::numeric_limits<double>::epsilon(),
				                      std::numeric_limits<double>::max());
			}
			return degreeGrid;
		}(
		)
	)
	, m_isRegression(isRegression)
{
}

void PolyKernel::calculateGrids(const BaseSvmChromosome& individual)
{
	m_cGrid = calculateNewGrid(m_cGrid, individual.getClassifier()->getC());
	m_degreeGrid = calculateNewGrid(m_degreeGrid, individual.getClassifier()->getGamma());
}

geneticComponents::Population<SvmKernelChromosome> PolyKernel::createGridSearchPopulation()
{
	std::vector<SvmKernelChromosome> population;
	auto steps = static_cast<int>(m_degreeGrid.maxVal - m_degreeGrid.minVal);

	for (auto i = 0u; i <= calculateNumberOfSteps(m_cGrid); ++i)
	{
		for (auto j = 0; j <= steps; ++j)
		{
			if (m_isRegression)
			{
				population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::Poly,
				                                            std::vector<double>{
					                                            m_cGrid.minVal * std::pow(m_cGrid.logStep, i),
					                                            m_degreeGrid.minVal + j,
					                                            0.001 //epsilon value
				                                            }, m_isRegression));
			}
			else
			{
				population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::Poly,
				                                            std::vector<double>{
																m_cGrid.minVal* std::pow(m_cGrid.logStep, i),
																m_degreeGrid.minVal + j,
																1
				                                            }, m_isRegression));
			}
		}
	}
	return geneticComponents::Population<SvmKernelChromosome>(population);
}

// void PolyKernel::calculateGrids()
// {
// 	m_cGrid = calculateNewGrid(m_cGrid, m_svm->getC());
// 	m_degreeGrid = calculateNewGrid(m_degreeGrid, m_svm->getGamma());
// }

// std::string PolyKernel::logSvmParameters()
// {
// 	return "Svm parameters(degree, C):\t" +
// 			std::to_string(m_svm->getDegree()) + "\t" +
// 			std::to_string(m_svm->getC());
// }

// void PolyKernel::performGridSearch(cv::Ptr<cv::ml::TrainData> /*trainingSet*/, unsigned /*numberOfFolds*/)
// {
// 	throw std::exception("Not implemented");
// 	//m_svm->trainAuto(trainingSet, numberOfFolds, m_cGrid, m_degreeGrid);
// }
} // namespace svmComponents
