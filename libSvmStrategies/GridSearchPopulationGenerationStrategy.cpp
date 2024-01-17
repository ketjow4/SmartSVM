#include "libSvmStrategies/GridSearchPopulationGenerationStrategy.h"
#include "libSvmComponents/SvmComponentsExceptions.h"

namespace svmStrategies
{
GridSearchPopulationGenerationStrategy::GridSearchPopulationGenerationStrategy(svmComponents::BaseKernelGridSearch& gridSearchKernel)
    : m_gridSearchKernel(gridSearchKernel)
{
}

std::string GridSearchPopulationGenerationStrategy::getDescription() const
{
    return "Generate population with parameters of grid search, can narrow grid on each run if best solution is provided";
}

geneticComponents::Population<svmComponents::SvmKernelChromosome> GridSearchPopulationGenerationStrategy::launch(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population)
{
    if (!population.empty())
    {
        m_gridSearchKernel.calculateGrids(population.getBestOne());
    }
    return m_gridSearchKernel.createGridSearchPopulation();
}
} // namespace svmStrategies
