
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/Population.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2012 Kawulok and Nalepa - Support vector machines training data selection using a genetic algorithm
class GaSvmGeneration : public geneticComponents::IPopulationGeneration<SvmTrainingSetChromosome>
{
public:
    explicit GaSvmGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                             std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                             unsigned int numberOfClassExamples,
                             const std::vector<unsigned int>& labelsCount);

    geneticComponents::Population<SvmTrainingSetChromosome> createPopulation(uint32_t populationSize) override;

private:
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClassExamples;
    unsigned int m_numberOfClasses;
};



class GaSvmGenerationWithForbbidenSet : public geneticComponents::IPopulationGeneration<SvmTrainingSetChromosome>
{
public:
	explicit GaSvmGenerationWithForbbidenSet(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                         std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
	                                         unsigned int numberOfClassExamples,
	                                         const std::vector<unsigned int>& labelsCount);

	geneticComponents::Population<SvmTrainingSetChromosome> createPopulation(uint32_t populationSize) override;

	void setNumberOfClassExamples(unsigned int newK)
	{
		m_numberOfClassExamples = newK;
	}

	void setTrainingSet(dataset::Dataset<std::vector<float>, float>& trainingSet)
	{
		m_trainingSet = trainingSet;
	}

	void setForbbidens(const std::unordered_set<uint64_t>& ids)
	{
		m_forbiddenIds = ids;
	}

	void setImbalancedOrOneClass(bool value)
	{
		m_imbalancedOrOneClass = value;
	}

private:
	dataset::Dataset<std::vector<float>, float> m_trainingSet;
	std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
	unsigned int m_numberOfClassExamples;
	unsigned int m_numberOfClasses;
	std::unordered_set<uint64_t> m_forbiddenIds;
	bool m_imbalancedOrOneClass;
};
} // namespace svmComponents
