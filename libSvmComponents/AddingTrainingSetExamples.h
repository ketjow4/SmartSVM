#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/TraningSetCrossover.h"
#include <unordered_set>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libDataset/Dataset.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libPlatform/Percent.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/Population.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"

namespace svmComponents
{
class ITrainingSet
{
public:
	virtual ~ITrainingSet() = default;
	virtual const dataset::Dataset<std::vector<float>, float>& trainingSet() = 0;
};


class FullTrainingSet : public ITrainingSet
{
public:
	FullTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet)
		: m_trainingSet(trainingSet)
	{
	}

	
	const dataset::Dataset<std::vector<float>, float>& trainingSet() override
	{
		return  m_trainingSet;
	}
	
private:
	const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
};


//ID bigger than original training set are the additional samples. Need to provide few function for that to make visualization easier
class SetupAdditionalVectors : public ITrainingSet
{
public:
	SetupAdditionalVectors(const dataset::Dataset<std::vector<float>, float>& trainingSet)
		: m_trainingSet(trainingSet)
	{}
	
	dataset::Dataset<std::vector<float>, float> getAdditionalVectors() const
	{
		return m_additionalVectors;
	}

	//need to keep the added samples UNIQUE !!!!
	void updateAdditionalVectors(dataset::Dataset<std::vector<float>, float>& modifiedDataset) 
	{
		m_additionalVectors = modifiedDataset;
		/*auto samples = modifiedDataset.getSamples();
		auto labels = modifiedDataset.getLabels();
		
		for(auto i = 0u; i < samples.size(); ++i)
		{
			m_additionalVectors.addSample(std::move(samples[i]), labels[i]);
		}*/
		
		updateCombinedSet();
	}

	const dataset::Dataset<std::vector<float>, float>& trainingSet() override
	{
		return m_combinedSet;
		//return  m_trainingSet + m_additionalVectors;
	}

private:
	void updateCombinedSet()
	{
		m_combinedSet = m_trainingSet;

		auto samples = m_additionalVectors.getSamples();
		auto labels = m_additionalVectors.getLabels();

		for (auto i = 0u; i < samples.size(); ++i)
		{
			m_combinedSet.addSample(samples[i], labels[i]);
		}
		//return modifiedTrainingSet;
		
	}

	
	 dataset::Dataset<std::vector<float>, float> m_additionalVectors;
	 const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
	 dataset::Dataset<std::vector<float>, float> m_combinedSet;
};


class CrossoverWithAdditionalExamples : public TrainingSetCrossover
{
public:
	using chromosomeType = SvmTrainingSetChromosome;

	explicit CrossoverWithAdditionalExamples(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
	                        unsigned int numberOfClasses,
		SetupAdditionalVectors& additinalVectors)
		:  TrainingSetCrossover(std::move(rngEngine), numberOfClasses)
	, m_additionalVectors(additinalVectors)
	{
	}

	chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& /*parentB*/) override
	{
		//std::cout << "crossover\n";
		return parentA;
	}

private:
	SetupAdditionalVectors& m_additionalVectors;
};


class GenerationWithAdditionalExamples : public geneticComponents::IPopulationGeneration<SvmTrainingSetChromosome>
{
public:
	explicit GenerationWithAdditionalExamples(const dataset::Dataset<std::vector<float>, float>& trainingSet,
					                          std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
					                          unsigned int numberOfClassExamples,
					                          const std::vector<unsigned int>& labelsCount,
											  SetupAdditionalVectors& additinalVectors)
		: m_trainingSet(trainingSet)
		, m_rngEngine(std::move(rngEngine))
		, m_numberOfClassExamples(numberOfClassExamples)
		, m_numberOfClasses(static_cast<unsigned int>(labelsCount.size()))
		, m_additionalVectors(additinalVectors)
	{
		if (std::any_of(labelsCount.begin(), labelsCount.end(), [this](const auto& labelCount) { return m_numberOfClassExamples > labelCount;  }))
		{
			//throw ValueOfClassExamplesIsTooHighForDataset(m_numberOfClassExamples);
		}
	}

	geneticComponents::Population<SvmTrainingSetChromosome> createPopulation(uint32_t populationSize) override
	{
		if (populationSize == 0)
		{
			throw geneticComponents::PopulationIsEmptyException();
		}
		
		dataset::Dataset<std::vector<float>, float> additional;
		additional.addSample(std::vector<float>{0.5, 0.5}, 1.0);
		m_additionalVectors.updateAdditionalVectors(additional);
		
		auto targets = m_trainingSet.getLabels();
		auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));
		std::vector<SvmTrainingSetChromosome> population(populationSize);

		std::generate(population.begin(), population.end(), [&]
			{
				std::unordered_set<std::uint64_t> trainingSet;
				std::vector<DatasetVector> chromosomeDataset;
				chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
				std::vector<unsigned int> classCount(m_numberOfClasses, 0);
				int j = 0;
				while (std::any_of(classCount.begin(), classCount.end(), [this](const auto& classIndicies) { return classIndicies != m_numberOfClassExamples;  }))
				{
					if(j == 0)
					{
						//single vector added
						chromosomeDataset.emplace_back(DatasetVector(m_trainingSet.size(), static_cast<std::uint8_t>(1)));
						classCount[1]++;
						j++;
						continue;
					}
					auto randomValue = m_rngEngine->getRandom(trainingSetID);
					if (classCount[static_cast<int>(targets[randomValue])] < m_numberOfClassExamples &&    // less that desired number of class examples
						trainingSet.emplace(static_cast<int>(randomValue)).second)       // is unique
					{
						chromosomeDataset.emplace_back(DatasetVector(randomValue, static_cast<std::uint8_t>(targets[randomValue])));
						classCount[static_cast<int>(targets[randomValue])]++;
					}
					j++;
				}

				
				return SvmTrainingSetChromosome(std::move(chromosomeDataset));
			});
		return geneticComponents::Population<SvmTrainingSetChromosome>(std::move(population));
	}

private:
	SetupAdditionalVectors& m_additionalVectors;
	const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
	std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	unsigned int m_numberOfClassExamples;
	unsigned int m_numberOfClasses;
};

// @wdudzik implementation based on: 2012 Kawulok and Nalepa - Support vector machines training data selection using a genetic algorithm
class MutationWithAdditionalExamples : public geneticComponents::IMutationOperator<SvmTrainingSetChromosome>
{
public:
	/*explicit MutationWithAdditionalExamples(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
	                       platform::Percent exchangePercent,
	                       platform::Percent mutationProbability,
	                       const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                       const std::vector<unsigned int>& labelsCount);*/

	explicit MutationWithAdditionalExamples(SetupAdditionalVectors& additionalVectors)
		: m_additionalVectors(additionalVectors)
	{
		
	}

	void mutatePopulation(geneticComponents::Population<SvmTrainingSetChromosome>& /*population*/) override
	{
		//std::cout << "mutation\n";
	}
	
	void mutateChromosome(SvmTrainingSetChromosome& /*chromosome*/) override
	{
		
	}

private:
	std::unordered_set<uint64_t> setDifference(const std::vector<DatasetVector>& set, const std::unordered_set<uint64_t>& deleted);
	void calculateNumberOfPossibleExchanges(SvmTrainingSetChromosome& chromosome,
	                                        std::vector<uint64_t>& possibleNumberOfExchangesPerClass) const;
	std::vector<DatasetVector> findReplacement(const std::unordered_set<uint64_t>& deleted,
	                                           std::unordered_set<uint64_t>& mutated,
	                                           const std::vector<std::size_t>& positionsToReplace,
	                                           SvmTrainingSetChromosome& chromosome) const;
	void getPositionsOfMutation(SvmTrainingSetChromosome& chromosome,
	                            std::unordered_set<uint64_t>& deleted,
	                            std::vector<std::size_t>& positionsToReplace) const;


	SetupAdditionalVectors& m_additionalVectors;
	/*std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	platform::Percent m_mutationProbability;
	platform::Percent m_exchangePercent;
	unsigned int m_numberOfExchanges;
	const unsigned int m_numberOfClasses;
	const std::vector<unsigned int> m_labelsCount;
	const dataset::Dataset<std::vector<float>, float>& m_trainingSet;*/
};
} // namespace svmComponents
