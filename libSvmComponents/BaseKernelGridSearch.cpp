
#include "BaseKernelGridSearch.h"

namespace svmComponents
{
ParamGrid BaseKernelGridSearch::calculateNewGrid(const ParamGrid& parametersGrid, double gridParemeter) const
{
    const auto numberOfSteps = calculateNumberOfSteps(parametersGrid);
    const auto newMin = gridParemeter / parametersGrid.logStep;
    const auto newMax = gridParemeter * parametersGrid.logStep;
    const auto rootIndex = 1.0 / numberOfSteps;
    const auto newLogStep = std::pow(newMax / newMin, rootIndex);
    return ParamGrid(newMin, newMax, newLogStep);
}

unsigned int BaseKernelGridSearch::calculateNumberOfSteps(const ParamGrid& parametersGrid)
{
    return static_cast<unsigned int>(std::floor(log(parametersGrid.maxVal / parametersGrid.minVal) / log(parametersGrid.logStep)));
}
} // namespace svmComponents