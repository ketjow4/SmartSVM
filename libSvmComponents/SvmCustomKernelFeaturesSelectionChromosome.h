#pragma once
#include "libSvmComponents/BaseSvmChromosome.h"
#include "Feature.h"
#include "SvmFeatureSetMemeticChromosome.h"
#include "SvmCustomKernelFeaturesSelectionChromosome.h"
#include "SvmCustomKernelChromosome.h"

namespace svmComponents
{

    class SvmCustomKernelFeaturesSelectionChromosome : public BaseSvmChromosome
    {
    public:
        SvmCustomKernelFeaturesSelectionChromosome() {}

        SvmCustomKernelFeaturesSelectionChromosome(SvmCustomKernelChromosome kernel,
            SvmFeatureSetMemeticChromosome features);

        
        SvmCustomKernelChromosome getKernel()
        {
            m_kernel.updateClassifier(this->getClassifier());
            m_kernel.updateConfusionMatrix(this->getConfusionMatrix());
            m_kernel.updateTime(this->getTime());
            m_kernel.updateFitness(this->getFitness());
            return m_kernel;
        }

        const std::vector<Gene>& getKernelDataset() const
        {
            return m_kernel.getDataset();
        }
    	
        dataset::Dataset<std::vector<float>, float> convertChromosome(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const;

    	
        const std::vector<Feature>& getFeatures() const;
        void updateFeatureSet(std::vector<Feature>& featureSet);
        std::unordered_set<std::uint64_t> convertFeaturesToSet() const;
    	
        dataset::Dataset<std::vector<float>, float> convertFeatures(const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const;


        void updateFitness(double fitness) override
        {
            m_fitness = fitness;
            m_kernel.updateFitness(fitness);
            m_features.updateFitness(fitness);
        }

        void updateClassifier(std::shared_ptr<phd::svm::ISvm> classifier) override
        {
            m_classifier = std::move(classifier);
            m_kernel.updateClassifier(m_classifier);
            m_features.updateClassifier(m_classifier);
        }

        SvmFeatureSetMemeticChromosome getFeaturesChromosome()
        {
            m_features.updateClassifier(this->getClassifier());
            m_features.updateConfusionMatrix(this->getConfusionMatrix());
            m_features.updateTime(this->getTime());
            m_features.updateFitness(this->getFitness());
            return m_features;
        }

        void setFeatures(const SvmFeatureSetMemeticChromosome& kernel)
        {
            m_features = kernel;
        }


        size_t trainingSetSize() const;
        size_t featureSetSize() const;
    private:
        SvmCustomKernelChromosome m_kernel;
        SvmFeatureSetMemeticChromosome m_features;
    };

    inline SvmCustomKernelFeaturesSelectionChromosome::SvmCustomKernelFeaturesSelectionChromosome(SvmCustomKernelChromosome kernel,
        SvmFeatureSetMemeticChromosome features)
        : m_kernel(kernel)
        , m_features(features)
    {
    }


    inline const std::vector<Feature>& SvmCustomKernelFeaturesSelectionChromosome::getFeatures() const
    {
        return m_features.getDataset();
    }

    inline void SvmCustomKernelFeaturesSelectionChromosome::updateFeatureSet(std::vector<Feature>& featureSet)
    {
        m_features.updateDataset(featureSet);
    }

    inline std::unordered_set<std::uint64_t> SvmCustomKernelFeaturesSelectionChromosome::convertFeaturesToSet() const
    {
        return m_features.convertToSet();
    }


    inline size_t SvmCustomKernelFeaturesSelectionChromosome::featureSetSize() const
    {
        return m_features.size();
    }

    inline dataset::Dataset<std::vector<float>, float> SvmCustomKernelFeaturesSelectionChromosome::convertChromosome(
        const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
    {
        const auto vectorsFiltered = m_kernel.convertChromosome(trainingDataSet);
        return m_features.convertChromosome(vectorsFiltered);
    }

    inline dataset::Dataset<std::vector<float>, float> SvmCustomKernelFeaturesSelectionChromosome::convertFeatures(
        const dataset::Dataset<std::vector<float>, float>& trainingDataSet) const
    {
        return m_features.convertChromosome(trainingDataSet);
    }
} // namespace svmComponents
