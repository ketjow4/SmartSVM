#pragma once

#include <random>

#include "libGeneticComponents/Population.h"
#include "libPlatform/Percent.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "libSvmComponents/ISvmMetricsCalculator.h"
#include "libSvmComponents/BaseSvmChromosome.h"
#include "libSvmComponents/SvmFeatureSetChromosome.h"
#include "libSvmComponents/SvmFeatureSetMemeticChromosome.h"
#include "libSvmComponents/SvmSimultaneousChromosome.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "SvmVisualization.h"
#include "libStrategies/FileSinkStrategy.h"
#include "LibGeneticSvm/WorkflowUtils.h"
#include "libSvmComponents/SvmComponentsExceptions.h"
#include "libSvmComponents/SvmUtils.h"

namespace svmStrategies
{
template <class chromosome>
class IValidationSubsetSelection
{
public:
	virtual ~IValidationSubsetSelection() = default;

	//Do not sort the population as the order of the individual should stay the same!!!!!
	virtual std::vector<dataset::Dataset<std::vector<float>, float>> selectValidationSubset(geneticComponents::Population<chromosome>& population,
	                                                                      const dataset::Dataset<std::vector<float>, float>& validationSet) = 0;
};

//select first 50 samples from validation set
template <class chromosome>
class DummySelection : public IValidationSubsetSelection<chromosome>
{
public:
	std::vector<dataset::Dataset<std::vector<float>, float>> selectValidationSubset(geneticComponents::Population<chromosome>& population,
		const dataset::Dataset<std::vector<float>, float>& validationSet) override
	{
		std::vector<dataset::Dataset<std::vector<float>, float>> validationSets;

		std::vector<svmComponents::DatasetVector> dummySet;
		auto labels = validationSet.getLabels();

		for (int i = 0; i < 50; ++i)
		{
			dummySet.emplace_back(i, labels[i]);
		}

		svmComponents::SvmTrainingSetChromosome s({}, std::move(dummySet) );

		
		for(auto i = 0; i < population.size(); ++i)
		{
			validationSets.emplace_back(s.convertValidationChromosome(validationSet));
			//example of how to update the validation set -- only works for SvmTrainingSetChromosome
			if constexpr (std::is_base_of<svmComponents::SvmTrainingSetChromosome, chromosome>::value)
			{
				population[i].updateValidationDataset(s.getValidationDataset());
			}
		}

		//TODO improve the visualization code
		bool doVisualization = true;
		if (doVisualization)
		{
			static int i = 0;
			svmComponents::SvmVisualization visualization2;
			auto image2 = visualization2.createVisualizationNewValidationSet(500, 500, validationSets[0]);
			strategies::FileSinkStrategy m_savePngElement;
			auto outputPath = genetic::generateFilenameWithTimestamp("validationSet.png", "demo_" + std::to_string(i) + "__", R"(C:\outputfolder)");
			m_savePngElement.launch(image2, outputPath);
			i++;
		}

		return validationSets;
	}
};


template <class chromosome>
class RandomSubsetPerIteration : public IValidationSubsetSelection<chromosome>
{
public:
	RandomSubsetPerIteration(platform::Percent subsetPercent,
		std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine)
		: m_subsetPercent(subsetPercent)
		, m_rngEngine(std::move(rngEngine))
	{
		
	}

	void generateSubset(const dataset::Dataset<std::vector<float>, float>& validationSet)
	{
		std::vector<svmComponents::DatasetVector> newSubset;
		auto labels = validationSet.getLabels();
		auto size = m_subsetPercent.m_percentValue * validationSet.size();
		auto validationSetID = std::uniform_int_distribution<int>(0, static_cast<int>(validationSet.size() - 1));
		std::unordered_set<std::uint64_t> vSet;

		while (vSet.size() < size)
		{
			auto randomValue = m_rngEngine->getRandom(validationSetID);
			if (vSet.emplace(static_cast<int>(randomValue)).second)       // is unique
			{
				newSubset.emplace_back(randomValue, static_cast<std::uint8_t>(labels[randomValue]));
			}
		}
		m_subset = newSubset;
	}
	
	std::vector<dataset::Dataset<std::vector<float>, float>> selectValidationSubset(geneticComponents::Population<chromosome>& /*population*/,
		const dataset::Dataset<std::vector<float>, float>& validationSet) override
	{
		std::vector<dataset::Dataset<std::vector<float>, float>> validationSets;

		auto newSubset = m_subset; //copy in here
		svmComponents::SvmTrainingSetChromosome s({}, std::move(newSubset));


		for (auto i = 0; i < 1; ++i)
		{
			validationSets.emplace_back(s.convertValidationChromosome(validationSet));
		}

		bool doVisualization = true;
		if (doVisualization)
		{
			static int i = 0;
			svmComponents::SvmVisualization visualization2;
			auto image2 = visualization2.createVisualizationNewValidationSet(500, 500, validationSets[0]);
			strategies::FileSinkStrategy m_savePngElement;
			auto outputPath = genetic::generateFilenameWithTimestamp("validationSet.png", "demo_" + std::to_string(i) + "__", R"(C:\outputfolder)");
			m_savePngElement.launch(image2, outputPath);
			i++;
		}

		return validationSets;
	}

private:
	platform::Percent m_subsetPercent;
	std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
	std::vector<svmComponents::DatasetVector> m_subset;
};


template <class chromosome>
class SvmSubsetValidation : public IValidationStrategy<chromosome>
{
	static_assert(std::is_base_of<svmComponents::BaseSvmChromosome, chromosome>::value, "Cannot do validation for class not derived from BaseSvmChromosome");
public:
	explicit SvmSubsetValidation(const svmComponents::ISvmMetricsCalculator& metric,
	                             std::shared_ptr<IValidationSubsetSelection<chromosome>> subsetSelectionMethod);

	geneticComponents::Population<chromosome>& launch(geneticComponents::Population<chromosome>& population,
	                                                  const dataset::Dataset<std::vector<float>, float>& validationSet) override;

	bool isUsingFullSet() const override { return false; }

	void generateNewSubset(const dataset::Dataset<std::vector<float>, float>& validationSet) override
	{
		reinterpret_cast<std::shared_ptr<RandomSubsetPerIteration<chromosome>>&>(m_subsetSelectionMethod)->generateSubset(validationSet);
	}

	
private:
	using Clock = std::chrono::high_resolution_clock;

	auto updateFitness(chromosome& individual,
	                   const dataset::Dataset<std::vector<float>, float>& validationSet);


	const svmComponents::ISvmMetricsCalculator& m_metric;
	std::shared_ptr<IValidationSubsetSelection<chromosome>> m_subsetSelectionMethod;
};

template <class chromosome>
SvmSubsetValidation<chromosome>::SvmSubsetValidation(const svmComponents::ISvmMetricsCalculator& metric,
                                                     std::shared_ptr<IValidationSubsetSelection<chromosome>> subsetSelectionMethod)
	: m_metric(metric)
	, m_subsetSelectionMethod(std::move(subsetSelectionMethod))
{
}

template <class chromosome>
geneticComponents::Population<chromosome>& SvmSubsetValidation<chromosome>::launch(geneticComponents::Population<chromosome>& population,
                                                                                   const dataset::Dataset<std::vector<float>, float>& validationSet)
{
	//in here code to modify validation set
	auto validationSets = m_subsetSelectionMethod->selectValidationSubset(population, validationSet); 
	//in here code to modify validation set

	for(auto& vSet : validationSets)
	{
		if(vSet.empty())
		{
			throw svmComponents::EmptyDatasetException(svmComponents::DatasetType::Validation);
		}
		auto labels = svmComponents::svmUtils::countLabels(2, vSet); //TODO handle mutliclass case in future
		if(labels[0] == 0 || labels[1] == 0)
		{
			throw svmComponents::OneClassValidationSet();
		}
	}
	
	
	const size_t iterationCount = std::distance(population.begin(), population.end());
	auto first = population.begin();

#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(iterationCount); i++)
	{
		auto& individual = *(first + i);
		if constexpr (std::is_base_of<svmComponents::SvmTrainingSetChromosome, chromosome>::value)
		{
			auto begin = updateFitness(individual, individual.convertValidationChromosome(validationSet));
			individual.updateTime(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin));
		}
		else if constexpr (std::is_base_of<svmComponents::SvmCustomKernelChromosome, chromosome>::value)
		{
			//NOTE: only single validation set here for whole poplation
			auto begin = updateFitness(individual, validationSets[0]); //version for index reference of individual
			individual.updateTime(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin));
		}
		else
		{
			auto begin = updateFitness(individual, validationSets[i]); //version for index reference of individual
			individual.updateTime(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin));
		}
		
	}

	return population;
}

template <class chromosome>
auto SvmSubsetValidation<chromosome>::updateFitness(chromosome& individual, const dataset::Dataset<std::vector<float>, float>& validationSet)
{
	auto begin = Clock::now();
	const auto metric = m_metric.calculateMetric(individual, validationSet);
	individual.updateFitness(metric.m_fitness);
	individual.updateConfusionMatrix(metric.m_confusionMatrix);
	individual.updateMetric(metric);
	return begin;
}

template <>
inline auto SvmSubsetValidation<svmComponents::SvmFeatureSetChromosome>::updateFitness(
	svmComponents::SvmFeatureSetChromosome& individual,
	const dataset::Dataset<std::vector<float>, float>& validationSet)
{
	const auto convertedSet = individual.convertChromosome(validationSet);
	const auto begin = Clock::now();
	const auto metric = m_metric.calculateMetric(individual, convertedSet);
	individual.updateFitness(metric.m_fitness);
	individual.updateConfusionMatrix(metric.m_confusionMatrix);
	individual.updateMetric(metric);
	return begin;
}

template <>
inline auto SvmSubsetValidation<svmComponents::SvmFeatureSetMemeticChromosome>::updateFitness(
	svmComponents::SvmFeatureSetMemeticChromosome& individual,
	const dataset::Dataset<std::vector<float>, float>& validationSet)
{
	const auto convertedSet = individual.convertChromosome(validationSet);
	const auto begin = Clock::now();
	const auto metric = m_metric.calculateMetric(individual, convertedSet);
	individual.updateFitness(metric.m_fitness);
	individual.updateConfusionMatrix(metric.m_confusionMatrix);
	individual.updateMetric(metric);
	return begin;
}

template <>
inline auto SvmSubsetValidation<svmComponents::SvmSimultaneousChromosome>::updateFitness(
	svmComponents::SvmSimultaneousChromosome& individual,
	const dataset::Dataset<std::vector<float>, float>& validationSet)
{
	const auto convertedSet = individual.convertFeatures(validationSet);
	const auto begin = Clock::now();
	const auto metric = m_metric.calculateMetric(individual, convertedSet);
	individual.updateFitness(metric.m_fitness);
	individual.updateConfusionMatrix(metric.m_confusionMatrix);
	individual.updateMetric(metric);
	return begin;
}
}
