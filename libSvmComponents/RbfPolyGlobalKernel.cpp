#include "RbfPolyGlobalKernel.h"
#include "libSvmComponents/SvmComponentsExceptions.h"

namespace svmComponents
{
RbfPolyGlobalKernel::RbfPolyGlobalKernel(cv::ml::ParamGrid cGrid, cv::ml::ParamGrid gammaGrid, cv::ml::ParamGrid degreeGrid, cv::ml::ParamGrid tGrid, bool isRegression)
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
		}())
		, m_tGrid([&tGrid]()
			{
				if (tGrid.minVal <= 0.0 || tGrid.maxVal >= 1.0)
				{
					throw ValueNotInRange("degreeGrid.minVal",
						tGrid.minVal,
						0.0 + std::numeric_limits<double>::epsilon(),
						tGrid.maxVal);
				}
				return tGrid;
			}())
	, m_isRegression(isRegression)
{
}

void RbfPolyGlobalKernel::calculateGrids(const BaseSvmChromosome& /*individual*/)
{
	throw std::exception("Not implemented");
	/*m_cGrid = calculateNewGrid(m_cGrid, individual.getClassifier()->getC());
	m_gammaGrid = calculateNewGrid(m_gammaGrid, individual.getClassifier()->getGamma());*/
}

geneticComponents::Population<SvmKernelChromosome> RbfPolyGlobalKernel::createGridSearchPopulation()
{
	std::vector<SvmKernelChromosome> population;
	const auto steps = static_cast<int>(m_degreeGrid.maxVal - m_degreeGrid.minVal);
	const auto tsteps = static_cast<int>(m_tGrid.maxVal * 10 - m_tGrid.minVal * 10);

	for (auto i = 0u; i <= calculateNumberOfSteps(m_cGrid); ++i)
	{
		for (auto j = 0u; j <= calculateNumberOfSteps(m_gammaGrid); ++j)
		{
			for(auto k = 0; k <= tsteps; k++)
			{
				for (auto l = 0; l <= steps; l++)
				{
					if (m_isRegression)
					{
						population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::RBF_POLY_GLOBAL,
							std::vector<double>{
							m_cGrid.minVal* std::pow(m_cGrid.logStep, i),
								m_gammaGrid.minVal* std::pow(m_gammaGrid.logStep, j),
								0.001, //epsilon value
								m_tGrid.minVal + 0.1 * k, //t value
								m_degreeGrid.minVal + l,
						}, m_isRegression));
					}
					else
					{
						population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::RBF_POLY_GLOBAL,
							std::vector<double>{
							m_cGrid.minVal* std::pow(m_cGrid.logStep, i),
								m_gammaGrid.minVal* std::pow(m_gammaGrid.logStep, j),
								m_tGrid.minVal + 0.1 * k, //t value
								m_degreeGrid.minVal + l,

						}, m_isRegression));
					}
				}
			}
		}
	}

	//for (auto i = 0u; i <= 2; ++i)
	//{
	//	for (auto j = 0u; j <= 1; ++j)
	//	{
	//		for (auto k = 0; k < 1; k++)
	//		{
	//			for (auto l = 0; l <= 1; l++)
	//			{
	//				if (m_isRegression)
	//				{
	//					population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::RBF_POLY_GLOBAL,
	//						std::vector<double>{
	//						m_cGrid.minVal* std::pow(m_cGrid.logStep, i),
	//							m_gammaGrid.minVal* std::pow(m_gammaGrid.logStep, j),
	//							0.001, //epsilon value
	//							m_tGrid.minVal + 0.1 * k, //t value
	//							m_degreeGrid.minVal + l,
	//					}, m_isRegression));
	//				}
	//				else
	//				{
	//					population.emplace_back(SvmKernelChromosome(phd::svm::KernelTypes::RBF_POLY_GLOBAL,
	//						std::vector<double>{
	//						m_cGrid.minVal* std::pow(m_cGrid.logStep, i),
	//							m_gammaGrid.minVal* std::pow(m_gammaGrid.logStep, j),
	//							m_tGrid.minVal + 0.1 * k, //t value
	//							m_degreeGrid.minVal + l,

	//					}, m_isRegression));
	//				}
	//			}
	//		}
	//	}
	//}
	return geneticComponents::Population<SvmKernelChromosome>(population);
}

void RbfPolyGlobalKernel::calculateGrids()
{
	m_cGrid = calculateNewGrid(m_cGrid, m_svm->getC());
	m_gammaGrid = calculateNewGrid(m_gammaGrid, m_svm->getGamma());
}

std::string RbfPolyGlobalKernel::logSvmParameters()
{
	return "Svm parameters(gamma, C):\t" +
			std::to_string(m_svm->getGamma()) + "\t" +
			std::to_string(m_svm->getC());
}

void RbfPolyGlobalKernel::performGridSearch(cv::Ptr<cv::ml::TrainData> /*trainingSet*/, unsigned /*numberOfFolds*/)
{
	throw std::exception("Not implemented");
	//m_svm->trainAuto(trainingSet, numberOfFolds, m_cGrid, m_gammaGrid);
}
} // namespace svmComponents
