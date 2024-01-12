
#pragma once

#include "SvmLib/SvmFactory.h"
#include "libSvmComponents/SvmConfigStructures.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "libSvmComponents/SvmUtils.h"
#include "libSvmComponents/SvmFeatureSetChromosome.h"
#include "libSvmComponents/ISvmTraining.h"
#include "SvmSimultaneousChromosome.h"

namespace svmComponents
{
template <class chromosome>
class SvmTraining : public ISvmTraining<chromosome>
{
public:
    explicit SvmTraining(const SvmAlgorithmConfiguration& svmConfig,
                         const SvmKernelChromosome& svmParameters,
                         bool probabilityOutputNeeded = false);

    void trainPopulation(geneticComponents::Population<chromosome>& population,
                         const dataset::Dataset<std::vector<float>, float>& trainingData) override;

    void updateParameters(const SvmKernelChromosome& parameters);
private:
    const SvmAlgorithmConfiguration m_svmConfig;
    const bool m_probabilityOutputNeeded;
    SvmKernelChromosome m_svmParameters;
    std::exception_ptr m_lastException;
};

template <class chromosome>
SvmTraining<chromosome>::SvmTraining(const SvmAlgorithmConfiguration& svmConfig,
                                     const SvmKernelChromosome& svmParameters,
                                     bool /*probabilityOutputNeeded*/)  
    : m_svmConfig(svmConfig)
    , m_probabilityOutputNeeded(false)  //19.10.2019 this is not needed as distance from hyperplane can be used as well
    , m_svmParameters(svmParameters)
{
}

template <class chromosome>
void SvmTraining<chromosome>::trainPopulation(geneticComponents::Population<chromosome>& population,
                                              const dataset::Dataset<std::vector<float>, float>& trainingData)
{
    if (population.empty())
    {
        throw geneticComponents::PopulationIsEmptyException();
    }
    if (trainingData.empty())
    {
        throw EmptyDatasetException(DatasetType::Training);
    }

    const size_t iterationCount = std::distance(population.begin(), population.end());
    auto first = population.begin();

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(iterationCount); i++)
    {
        try
        {
            auto& individual = *(first + i);
            auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);

            svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
            svmUtils::setupSvmParameters(*classifier, m_svmParameters);

            auto individualTraningSet = individual.convertChromosome(trainingData);
            if constexpr (std::is_same<chromosome, SvmTrainingSetChromosome>::value)
            {
                //this may not work correctly for FSALMA and KTF
                std::vector<Feature> vec;
                vec.resize(static_cast<int>(trainingData.getSample(0).size()));
                std::uint64_t j = 0;
                std::iota(vec.begin(), vec.end(), j);
                classifier->setFeatureSet(vec, static_cast<int>(trainingData.getSample(0).size()));
            }
            if constexpr (std::is_same<chromosome, SvmFeatureSetMemeticChromosome>::value)
            {
                classifier->setFeatureSet(individual.getDataset(), static_cast<int>(trainingData.getSample(0).size()));
            }
            classifier->train(individualTraningSet, m_probabilityOutputNeeded);
            individual.updateClassifier(std::move(classifier));
            
        }
        catch (const UnsupportedKernelTypeException& exception)
        {
#pragma omp critical
            m_lastException = std::make_exception_ptr(exception);
        }
    }
    if (m_lastException)
    {
        std::rethrow_exception(m_lastException);
    }
}

template <class chromosome>
void SvmTraining<chromosome>::updateParameters(const SvmKernelChromosome& parameters)
{
    m_svmParameters = parameters;
}

template
SvmTraining<SvmTrainingSetChromosome>;

template
SvmTraining<SvmFeatureSetChromosome>;





class SvmTrainingSSVM : public ISvmTraining<SvmSimultaneousChromosome>
{
public:
    explicit SvmTrainingSSVM(const SvmAlgorithmConfiguration& svmConfig,
                             bool probabilityOutputNeeded = false);

    void trainPopulation(geneticComponents::Population<SvmSimultaneousChromosome>& population,
                         const dataset::Dataset<std::vector<float>, float>& trainingData) override;

private:
    const SvmAlgorithmConfiguration m_svmConfig;
    const bool m_probabilityOutputNeeded;    
    std::exception_ptr m_lastException;
};

inline SvmTrainingSSVM::SvmTrainingSSVM(const SvmAlgorithmConfiguration& svmConfig, bool /*probabilityOutputNeeded*/)
    : m_svmConfig(svmConfig)
    , m_probabilityOutputNeeded(false)
{
}

inline void SvmTrainingSSVM::trainPopulation(geneticComponents::Population<SvmSimultaneousChromosome>& population,
                                             const dataset::Dataset<std::vector<float>, float>& trainingData)
{
    if (population.empty())
    {
        throw geneticComponents::PopulationIsEmptyException();
    }
    if (trainingData.empty())
    {
        throw EmptyDatasetException(DatasetType::Training);
    }

    const size_t iterationCount = std::distance(population.begin(), population.end());
    auto first = population.begin();

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(iterationCount); i++)
    {
        try
        {
            auto& individual = *(first + i);
            auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);

            svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
            svmUtils::setupSvmParameters(*classifier, individual.getKernel());

            auto individualTraningSet = individual.convertChromosome(trainingData);
            classifier->train(individualTraningSet, m_probabilityOutputNeeded);
            classifier->setFeatureSet(individual.getFeatures(), static_cast<int>(trainingData.getSample(0).size()));
            individual.updateClassifier(std::move(classifier));
        }
        catch (const UnsupportedKernelTypeException& exception)
        {
#pragma omp critical
            m_lastException = std::make_exception_ptr(exception);
        }
    }
    if (m_lastException)
    {
        std::rethrow_exception(m_lastException);
    }
}
} // namespace svmComponents
