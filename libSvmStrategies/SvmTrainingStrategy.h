#pragma once

#include "libGeneticComponents/GeneticExceptions.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "libSvmComponents/ISvmTraining.h"

namespace svmStrategies
{
template <class chromosome>
class SvmTrainingStrategy
{
    static_assert(std::is_base_of<svmComponents::BaseSvmChromosome, chromosome>::value, "Cannot do validation for class not derived from BaseSvmChromosome");
public:
    explicit SvmTrainingStrategy(svmComponents::ISvmTraining<chromosome>& m_trainingMethod);

    std::string getDescription() const;
    geneticComponents::Population<chromosome> launch(geneticComponents::Population<chromosome>& population,
                                                     const dataset::Dataset<std::vector<float>, float>& trainingDataset);

private:
    svmComponents::ISvmTraining<chromosome>& m_trainingMethod;
};

template <class chromosome>
SvmTrainingStrategy<chromosome>::SvmTrainingStrategy(svmComponents::ISvmTraining<chromosome>& trainingMethod)
    : m_trainingMethod(trainingMethod)
{
}

template <class chromosome>
std::string SvmTrainingStrategy<chromosome>::getDescription() const
{
    return "Train Svm classifiers for provided population";
}

template <class chromosome>
geneticComponents::Population<chromosome> SvmTrainingStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population,
                                                                                  const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
    /*try
    {*/
    m_trainingMethod.trainPopulation(population, trainingDataset);

    return population;
    /*}
    catch (const geneticComponents::PopulationIsEmptyException& exception)
    {
        handleException(exception);
    }
    catch (const svmComponents::EmptyDatasetException& exception)
    {
        handleException(exception);
    }
    catch (...)
    {
        m_status = framework::ElementStrategyStatus::Error;
        m_logger.LOG(logger::LogLevel::Error, "Unknown error in SvmTrainingStrategy");
    }*/
}
} // namespace svmStrategies
