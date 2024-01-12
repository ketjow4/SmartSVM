#pragma once

#include <unordered_set>
#include <gsl/span>
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/SvmFeatureSetMemeticChromosome.h"
#include "IFeatureSelection.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticFeaturesPool : public IFeatureSelection
{
public:
    MemeticFeaturesPool() = default;
    void updateFeaturesVectorPool(const geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                                  const dataset::Dataset<std::vector<float>, float>& trainingSet);

    void addFeatures(const SvmFeatureSetMemeticChromosome& chromosome, const dataset::Dataset<std::vector<float>, float>& trainingSet) override;
    const std::vector<Feature>& getFeaturesPool() const override;
    const std::unordered_set<uint64_t>& getFeaturesIds() const override;

private:
    /*static unsigned int findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
                                                    gsl::span<const float> supportVector);*/

    std::vector<Feature> m_featurePool;
    std::unordered_set<uint64_t> m_featuresIds;
};

inline void MemeticFeaturesPool::updateFeaturesVectorPool(const geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                                                          const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    //addFeatures(population.getBestOne(), trainingSet);
    /*m_featurePool.clear();
    m_featuresIds.clear();*/
    auto size = trainingSet.getSample(0).size();
    std::vector<int> histogram (static_cast<int>(size), 0 );
    for (const auto& individual : population)
    {
        auto& dataset = individual.getDataset();
        for(auto& f : dataset)
        {
            histogram[f.id]++;
        }
    }

    int i = 0;
    for(auto feature : histogram)
    {
        if(feature >= 4)
        {
            if (m_featuresIds.emplace(i).second)
            {
                m_featurePool.emplace_back(Feature(i));
            }
        }
        ++i;
    }

	if(m_featurePool.empty())
	{
        auto avgFitness = 0.0;
        for (const auto& individual : population)
        {
            avgFitness += individual.getFitness();
        }
        avgFitness /= population.size();
		
		for (const auto& individual : population)
		{
			if(individual.getFitness() >= avgFitness)
			{
				addFeatures(individual, trainingSet);
			}
		}
	}
}

inline void MemeticFeaturesPool::addFeatures(const SvmFeatureSetMemeticChromosome& chromosome, const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/)
{
    const auto classifier = chromosome.getClassifier();
    if (classifier && classifier->isTrained())
    {
        auto& dataset = chromosome.getDataset();

        for (auto i = 0; i < dataset.size(); i++)
        {
            if (m_featuresIds.emplace(dataset[i].id).second)
            {
                m_featurePool.emplace_back(Feature(dataset[i].id));
            }
        }
        return;
    }
    throw UntrainedSvmClassifierException();
}

inline const std::vector<Feature>& MemeticFeaturesPool::getFeaturesPool() const
{
    return m_featurePool;
}

inline const std::unordered_set<uint64_t>& MemeticFeaturesPool::getFeaturesIds() const
{
    return m_featuresIds;
}

//inline unsigned MemeticFeaturesPool::findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
//                                                                 gsl::span<const float> supportVector)
//{
//    auto samples = individualDataset.getSamples();
//    auto positionInDataset = std::find_if(samples.begin(),
//                                          samples.end(),
//                                          [&supportVector](const auto& sample)
//                                          {
//                                              return std::equal(sample.begin(),
//                                                                sample.end(),
//                                                                supportVector.begin(),
//                                                                supportVector.end());
//                                          }) - samples.begin();
//    return static_cast<unsigned int>(positionInDataset);
//}
} // namespace svmComponents
