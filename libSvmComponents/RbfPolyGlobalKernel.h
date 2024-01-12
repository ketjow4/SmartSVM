
#pragma once

#include "libSvmComponents/BaseKernelGridSearch.h"

namespace svmComponents
{
    class RbfPolyGlobalKernel : public BaseKernelGridSearch
    {
    public:
        RbfPolyGlobalKernel(cv::ml::ParamGrid cGrid, cv::ml::ParamGrid gammaGrid, cv::ml::ParamGrid degreeGrid, cv::ml::ParamGrid tGrid, bool isRegression);

        void calculateGrids(const BaseSvmChromosome& individual) override;
        geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() override;

        void calculateGrids() override;
        std::string logSvmParameters() override;
        void performGridSearch(cv::Ptr<cv::ml::TrainData> trainingSet, unsigned numberOfFolds) override;

    private:
        cv::ml::ParamGrid m_cGrid;
        cv::ml::ParamGrid m_gammaGrid;
        cv::ml::ParamGrid m_degreeGrid;
        cv::ml::ParamGrid m_tGrid;
        bool m_isRegression;
    };
} // namespace svmComponents
