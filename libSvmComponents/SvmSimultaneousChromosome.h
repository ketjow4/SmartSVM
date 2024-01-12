#pragma once
#include "libSvmComponents/BaseSvmChromosome.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "Feature.h"
#include "SvmKernelChromosome.h"
#include "SvmFeatureSetMemeticChromosome.h"
#include "SvmLib/OpenCvSvm.h"

namespace svmComponents
{
class ISvmCompleteChromosome
{
public:
    virtual ~ISvmCompleteChromosome() = default;
    //kernel
    virtual phd::svm::KernelTypes getKernelType() const = 0;
    virtual const std::vector<double>& getKernelParameters() const = 0;
    virtual void updateKernelParameters(std::vector<double>& parameters) = 0;
    virtual void updateKernelType(phd::svm::KernelTypes kernel) = 0;

    //training set
    virtual const std::vector<DatasetVector>& getDataset() const = 0;
    virtual void updateFeatureSet(std::vector<DatasetVector>& traningSet) = 0;
    virtual std::unordered_set<std::uint64_t> convertToSet() const = 0;
    virtual size_t trainingSetSize() const = 0;
    
    //feature set
    virtual const std::vector<Feature>& getFeatures() const = 0;
    virtual void updateFeatureSet(std::vector<Feature>& featureSet) = 0;
    virtual std::unordered_set<std::uint64_t> convertFeaturesToSet() const = 0;
    virtual size_t featureSetSize() const = 0;

    //conversion
    virtual dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const = 0;
    virtual dataset::Dataset<std::vector<float>, float> convertFeatures(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const = 0;
};


class SvmSimultaneousChromosome : public BaseSvmChromosome, public ISvmCompleteChromosome
{
public:
    SvmSimultaneousChromosome() {}

    SvmSimultaneousChromosome(SvmKernelChromosome kernel,
                              SvmTrainingSetChromosome training,
                              SvmFeatureSetMemeticChromosome features);

    phd::svm::KernelTypes getKernelType() const override;
    const std::vector<double>& getKernelParameters() const override;
    void updateKernelParameters(std::vector<double>& parameters) override;
    void updateKernelType(phd::svm::KernelTypes kernel) override;
    const std::vector<DatasetVector>& getDataset() const override;
    void updateFeatureSet(std::vector<DatasetVector>& traningSet) override;
    dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const override;
    std::unordered_set<std::uint64_t> convertToSet() const override;
    const std::vector<Feature>& getFeatures() const override;
    void updateFeatureSet(std::vector<Feature>& featureSet) override;
    dataset::Dataset<std::vector<float>, float> convertFeatures(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const override;
    std::unordered_set<std::uint64_t> convertFeaturesToSet() const override;

    void updateFitness(double fitness) override
    {
        m_fitness = fitness;
        m_kernel.updateFitness(fitness);
        m_features.updateFitness(fitness);
        m_training.updateFitness(fitness);
    }

    void updateClassifier(std::shared_ptr<phd::svm::ISvm> classifier) override
    {
        m_classifier = std::move(classifier);        
        m_kernel.updateClassifier(m_classifier);
        m_features.updateClassifier(m_classifier);
        m_training.updateClassifier(m_classifier);
    }

    SvmKernelChromosome getKernel() const
    {
        return m_kernel;
    }

    void setKernel(const SvmKernelChromosome& kernel)
    {
        m_kernel = kernel;
    }

    SvmFeatureSetMemeticChromosome getFeaturesChromosome() const
    {
        return m_features;
    }

    void setFeatures(const SvmFeatureSetMemeticChromosome& kernel)
    {
        m_features = kernel;
    }

    SvmTrainingSetChromosome getTraining() const
    {
        return m_training;
    }

    void setTraining(const SvmTrainingSetChromosome& kernel)
    {
        m_training = kernel;
    }

    size_t trainingSetSize() const override;
    size_t featureSetSize() const override;
private:
    SvmKernelChromosome m_kernel;
    SvmTrainingSetChromosome m_training;
    SvmFeatureSetMemeticChromosome m_features;
};

inline SvmSimultaneousChromosome::SvmSimultaneousChromosome(SvmKernelChromosome kernel, 
                                                            SvmTrainingSetChromosome training,
                                                            SvmFeatureSetMemeticChromosome features)
    : m_kernel(kernel)
    , m_training(training)
    , m_features(features)
{
}

inline phd::svm::KernelTypes SvmSimultaneousChromosome::getKernelType() const
{
    return m_kernel.getKernelType();
}

inline const std::vector<double>& SvmSimultaneousChromosome::getKernelParameters() const
{
    return m_kernel.getKernelParameters();
}

inline void SvmSimultaneousChromosome::updateKernelParameters(std::vector<double>& parameters)
{
    m_kernel.updateKernelParameters(parameters);
}

inline void SvmSimultaneousChromosome::updateKernelType(phd::svm::KernelTypes /*kernel*/)
{
    //TODO
    //m_kernel.
}

inline const std::vector<DatasetVector>& SvmSimultaneousChromosome::getDataset() const
{
    return m_training.getDataset();
}

inline void SvmSimultaneousChromosome::updateFeatureSet(std::vector<DatasetVector>& traningSet)
{
    m_training.updateDataset(traningSet);
}

inline std::unordered_set<std::uint64_t> SvmSimultaneousChromosome::convertToSet() const
{
    return m_training.convertToSet();
}

inline const std::vector<Feature>& SvmSimultaneousChromosome::getFeatures() const
{
    return m_features.getDataset();
}

inline void SvmSimultaneousChromosome::updateFeatureSet(std::vector<Feature>& featureSet)
{
    m_features.updateDataset(featureSet);
}

inline std::unordered_set<std::uint64_t> SvmSimultaneousChromosome::convertFeaturesToSet() const
{
    return m_features.convertToSet();
}

inline size_t SvmSimultaneousChromosome::trainingSetSize() const
{
    return m_training.size();
}

inline size_t SvmSimultaneousChromosome::featureSetSize() const
{
    return m_features.size();
}

inline dataset::Dataset<std::vector<float>, float> SvmSimultaneousChromosome::convertChromosome(
    const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
{
    const auto vectorsFiltered = m_training.convertChromosome(trainingDataSet);
    return m_features.convertChromosome(vectorsFiltered);
}

inline dataset::Dataset<std::vector<float>, float> SvmSimultaneousChromosome::convertFeatures(
    const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
{
    return m_features.convertChromosome(trainingDataSet);
}
} // namespace svmComponents
