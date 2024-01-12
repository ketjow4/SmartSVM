
#include "BaseKernelGridSearch.h"

namespace svmComponents
{
cv::ml::ParamGrid BaseKernelGridSearch::calculateNewGrid(const cv::ml::ParamGrid& parametersGrid, double gridParemeter) const
{
    const auto numberOfSteps = calculateNumberOfSteps(parametersGrid);
    const auto newMin = gridParemeter / parametersGrid.logStep;
    const auto newMax = gridParemeter * parametersGrid.logStep;
    const auto rootIndex = 1.0 / numberOfSteps;
    const auto newLogStep = std::pow(newMax / newMin, rootIndex);
    return cv::ml::ParamGrid(newMin, newMax, newLogStep);
}

unsigned int BaseKernelGridSearch::calculateNumberOfSteps(const cv::ml::ParamGrid& parametersGrid)
{
    return static_cast<unsigned int>(std::floor(log(parametersGrid.maxVal / parametersGrid.minVal) / log(parametersGrid.logStep)));
}
} // namespace svmComponents