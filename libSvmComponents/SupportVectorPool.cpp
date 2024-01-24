
#include <algorithm>
//#include <opencv2/core.hpp>
#include "SvmComponentsExceptions.h"
#include "SupportVectorPool.h"

#include "libPlatform/loguru.hpp"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
void SupportVectorPool::updateSupportVectorPool(const geneticComponents::Population<SvmTrainingSetChromosome>& population,
                                                const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    for (const auto& individual : population)
    {
        addSupportVectors(individual, trainingSet);
    }
}

unsigned int SupportVectorPool::findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
                                                            gsl::span<const float> supportVector) 
{
    auto samples = individualDataset.getSamples();
    auto positionInDataset = std::find_if(samples.begin(),
                                          samples.end(),
                                          [&supportVector](const auto& sample)
                                  {
                                      return std::equal(sample.begin(),
                                                        sample.end(),
                                                        supportVector.begin(),
                                                        supportVector.end());
                                  }) - samples.begin();
    return static_cast<unsigned int>(positionInDataset);
}

void SupportVectorPool::addSupportVectors(const SvmTrainingSetChromosome& chromosome,
                                          const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    const auto classifier = chromosome.getClassifier();
    if (classifier && classifier->isTrained())
    {
        auto supportVectors = classifier->getSupportVectors();
        auto individualDataset = chromosome.convertChromosome(trainingSet);
        
        if(!classifier->getFeatureSet().empty())
        {
            auto featureSet = classifier->getFeatureSet();
            SvmFeatureSetMemeticChromosome c(std::move(featureSet));
            individualDataset = c.convertChromosome(individualDataset);
        }

        auto& dataset = chromosome.getDataset();

        for (auto i = 0; i < supportVectors.size(); i++)
        {
            //const float* sv = supportVectors.ptr<float>(i);
            const gsl::span<const float> supportVector(supportVectors[i].data(), supportVectors[i].size());

            const auto positionInDataset = findPositionOfSupprotVector(individualDataset, supportVector);

            if(positionInDataset == individualDataset.size())
            {
                //LOG_F(INFO, "Support vector not found in individual, can happen with adding vectors from frozen Pool of SV into training, using DA-SVM kernels");
                continue;
            }
        	
            if (m_supportVectorIds.emplace(dataset[positionInDataset].id).second)
            {
                m_supportVectorPool.emplace_back(dataset[positionInDataset].id, dataset[positionInDataset].classValue);
            }
        }
        return;
    }
    throw UntrainedSvmClassifierException();
}
} // namespace svmComponents
