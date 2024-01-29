
#pragma once

#include "libSvmComponents/BaseKernelGridSearch.h"

namespace svmComponents
{
	class PolyKernel : public BaseKernelGridSearch
	{
	public:
		PolyKernel(ParamGrid cGrid, ParamGrid degreeGrid, bool isRegression);

		void calculateGrids(const BaseSvmChromosome& individual) override;
		geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() override;

		// void calculateGrids() override;
		// std::string logSvmParameters() override;
		// void performGridSearch(cv::Ptr<cv::ml::TrainData> trainingSet, unsigned numberOfFolds) override;

	private:
		ParamGrid m_cGrid;
		ParamGrid m_degreeGrid;
		bool m_isRegression;
		//logger::LogFrontend m_logger;
	};
} // namespace svmComponents
