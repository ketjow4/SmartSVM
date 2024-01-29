
#pragma once

#include "libSvmComponents/BaseKernelGridSearch.h"

namespace svmComponents
{
    class RbfPolyGlobalKernel : public BaseKernelGridSearch
    {
    public:
        RbfPolyGlobalKernel(ParamGrid cGrid, ParamGrid gammaGrid, ParamGrid degreeGrid, ParamGrid tGrid, bool isRegression);

        void calculateGrids(const BaseSvmChromosome& individual) override;
        geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() override;

        // void calculateGrids() override;
        // std::string logSvmParameters() override;
        // void performGridSearch(cv::Ptr<TrainData> trainingSet, unsigned numberOfFolds) override;

    private:
        ParamGrid m_cGrid;
        ParamGrid m_gammaGrid;
        ParamGrid m_degreeGrid;
        ParamGrid m_tGrid;
        bool m_isRegression;
    };
} // namespace svmComponents
