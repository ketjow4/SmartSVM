
#include "SvmLib/OpenCvSvm.h"
#include "libGeneticComponents/GeneticExceptions.h"
#include "SvmKernelTraining.h"
#include "SvmUtils.h"
#include "SvmComponentsExceptions.h"

namespace svmComponents
{
SvmKernelTraining::SvmKernelTraining(const SvmAlgorithmConfiguration& svmConfig, bool /*probabilityNeeded*/)
    : m_svmConfig(svmConfig)
    , m_probabilityNeeded(false)
{
}

void SvmKernelTraining::trainPopulation(geneticComponents::Population<SvmKernelChromosome>& population,
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

#pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < static_cast<int>(iterationCount); i++)
    {
        try
        {
            auto& individual = *(first + i);
            auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);
			//auto classifier = phd::svm::SvmFactory::create(phd::svm::SvmImplementationType::OpenCvSvm);

            svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
            svmUtils::setupSvmParameters(*classifier, individual);
            classifier->train(trainingData, m_probabilityNeeded);

            //TODO wrong results when using FSALMA or KTF
            std::vector<Feature> vec;
            vec.resize(static_cast<int>(trainingData.getSample(0).size()));
            std::uint64_t j = 0;
            std::iota(vec.begin(), vec.end(), j);
            classifier->setFeatureSet(vec, static_cast<int>(trainingData.getSample(0).size()));

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
