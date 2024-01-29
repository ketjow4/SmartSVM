
#pragma once

#include "libSvmComponents/BaseKernelGridSearch.h"

namespace svmComponents
{
class RbfKernel : public BaseKernelGridSearch
{
public:
    RbfKernel(ParamGrid cGrid, ParamGrid gammaGrid, bool isRegression);

    void calculateGrids(const BaseSvmChromosome& individual) override;
    geneticComponents::Population<SvmKernelChromosome> createGridSearchPopulation() override;

    // void calculateGrids() override;
    // std::string logSvmParameters() override;
    // void performGridSearch(cv::Ptr<TrainData> trainingSet, unsigned numberOfFolds) override;

private:
    ParamGrid m_cGrid;
    ParamGrid m_gammaGrid;
    bool m_isRegression;
    //logger::LogFrontend m_logger;
};
} // namespace svmComponents
