
#pragma once

#include "libSvmComponents/BaseKernelGridSearch.h"

namespace svmComponents
{
	class LinearKernel : public BaseKernelGridSearch
	{
	public:
		LinearKernel(cv::ml::ParamGrid cGrid, bool isRegression);

		void calculateGrids(const BaseSvmChromosome& individual) override;
		geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() override;

		void calculateGrids() override;
		std::string logSvmParameters() override;
		void performGridSearch(cv::Ptr<cv::ml::TrainData> trainingSet, unsigned numberOfFolds) override;

	private:
		cv::ml::ParamGrid m_cGrid;
		bool m_isRegression;
	};
} // namespace svmComponents
