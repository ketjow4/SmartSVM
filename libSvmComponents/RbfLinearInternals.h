#pragma once

#include "LibGeneticComponents/IPopulationGeneration.h"
#include "SvmCustomKernelChromosome.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "libPlatform/Percent.h"
#include "ISupportVectorSelection.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
//#include "MemeticTraningSetAdaptation.h"
#include "libSvmComponents/SvmUtils.h"
#include "libGeneticComponents/IMutationOperator.h"

namespace svmComponents
{
class CusomKernelGenerationRbfLinearSequential : public geneticComponents::IPopulationGeneration<SvmCustomKernelChromosome>
{
public:
	explicit CusomKernelGenerationRbfLinearSequential(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                                                  std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
	                                                  unsigned int numberOfClassExamples,
	                                                  const std::vector<unsigned int>& labelsCount);

	geneticComponents::Population<SvmCustomKernelChromosome> createPopulation(uint32_t populationSize) override;

	void setCandGamma(std::vector<double>& C, std::vector<double>& gamma)
	{
		m_C = C;
		m_gamma = gamma;
	}

	void setCandGammaSingle(double C, double gamma)
	{
		m_CSingle = C;
		m_gammaSingle = gamma;
	}

	void setGamma(double gamma)
	{
		m_gammaSingle = gamma;
	}

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
	std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	unsigned int m_numberOfClassExamples;
	unsigned int m_numberOfClasses;
	std::vector<double> m_C;
	std::vector<double> m_gamma;
	double m_gammaSingle;
	double m_CSingle;
	std::unordered_set<uint64_t> m_forbiddenIds;
	bool m_imbalancedOrOneClass;
};

class CrossoverCompensationRbfLinear
{
public:
	CrossoverCompensationRbfLinear(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                               std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
	                               unsigned int numberOfClasses);

	geneticComponents::Population<SvmCustomKernelChromosome> compensate(geneticComponents::Population<SvmCustomKernelChromosome>& population,
	                                                                    const std::vector<unsigned int>& compensationInfo);

	void setGamma(double gamma)
	{
		m_gamma = gamma;
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
	const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
	std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	unsigned int m_numberOfClasses;
	double m_gamma;
	std::unordered_set<uint64_t> m_forbiddenIds;
	bool m_imbalancedOrOneClass;
};

//
class EducationOfTrainingSetRbfLinear
{
public:
	virtual ~EducationOfTrainingSetRbfLinear() = default;

	EducationOfTrainingSetRbfLinear(platform::Percent educationProbability,
	                                unsigned int numberOfClasses,
	                                std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
	                                std::unique_ptr<ISupportVectorSelectionGamma> supportVectorSelection);

	void educatePopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population,
	                       const std::vector<Gene>& supportVectorPool,
	                       const std::vector<geneticComponents::Parents<SvmCustomKernelChromosome>>& parents,
	                       const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

	void educate(SvmCustomKernelChromosome& individual,
	             const geneticComponents::Parents<SvmCustomKernelChromosome>& parents,
	             const std::vector<Gene>& supportVectorPool,
	             const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

private:
	static bool replacementCondition(const Gene& supportVectorPoolElement,
	                                 std::unordered_set<std::uint64_t>& traningIDs,
	                                 const Gene& sample);

	static std::vector<Gene> setDifference(const std::unordered_set<uint64_t>& svPool,
	                                       const std::vector<Gene>& traningDataset);

	std::unordered_set<uint64_t> findSupportVectors(const geneticComponents::Parents<SvmCustomKernelChromosome>& parents,
	                                                const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

	platform::Percent m_educationProbability;
	const unsigned int m_numberOfClasses;
	std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	std::unique_ptr<ISupportVectorSelectionGamma> m_supportVectorSelection;
};

class SupportVectorPoolRbfLinear : public ISupportVectorSelectionGamma
{
public:
	SupportVectorPoolRbfLinear() = default;
	void updateSupportVectorPool(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
	                             const dataset::Dataset<std::vector<float>, float>& trainingSet);

	void addSupportVectors(const SvmCustomKernelChromosome& chromosome,
	                       const dataset::Dataset<std::vector<float>, float>& trainingSet) override;

	const std::vector<Gene>& getSupportVectorPool() const override;
	const std::unordered_set<uint64_t>& getSupportVectorIds() const override;

	void clear()
	{
		m_supportVectorPool.clear();
		m_supportVectorIds.clear();
	}

	void setCurrentGamma(double gamma)
	{
		m_currentGamma = gamma;
	}

private:
	static unsigned int findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
	                                                gsl::span<const float> supportVector);

	std::vector<Gene> m_supportVectorPool;
	std::unordered_set<uint64_t> m_supportVectorIds;
	double m_currentGamma;
};

class SuperIndividualsCreationRbfLinear
{
public:
	SuperIndividualsCreationRbfLinear(std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
	                                  unsigned int numberOfClasses);

	geneticComponents::Population<SvmCustomKernelChromosome> createPopulation(uint32_t populationSize,
	                                                                          const std::vector<Gene>& supportVectorPool,
	                                                                          unsigned int numberOfClassExamples);

	void setImbalancedOrOneClass(bool value)
	{
		m_imbalancedOrOneClass = value;
	}

	void setC(double c)
	{
		m_Cvalue = c;
	}

private:
	std::vector<SvmCustomKernelChromosome> generate(unsigned int populationSize,
	                                                const std::vector<Gene>& supportVectorPool,
	                                                unsigned int numberOfClassExamples);

	const std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	const unsigned int m_numberOfClasses;
	bool m_imbalancedOrOneClass;
	double m_Cvalue;
};

using namespace geneticComponents;

class MemeticTrainingSetAdaptationRbfLinear
{
public:
	MemeticTrainingSetAdaptationRbfLinear(bool isLocalMode,
	                                      unsigned int numberOfClassExamples,
	                                      platform::Percent percentOfSupportVectorsThreshold,
	                                      unsigned int iterationsBeforeChangeThreshold,
	                                      const std::vector<unsigned int>& classCount,
	                                      double thresholdForMaxNumberOfClassExamples);
	void adapt(geneticComponents::Population<SvmCustomKernelChromosome>& population);

	bool getIsModeLocal() const;
	unsigned int getNumberOfClassExamples() const;

	void resetToInitial(unsigned int numberOfClassExamples);

	void setFrozenSetSize(unsigned int size);

private:
	void validate() const;
	void growSizeOfTraningSet(double bestOneFitness, platform::Percent percentOfSupportVectors);
	bool adaptationCondition(double deltaIteration,
	                         double deltaMode,
	                         platform::Percent percentOfSupportVectors,
	                         double improvementRate) const;

	bool m_isLocalMode;
	unsigned int m_numberOfClassExamples;
	unsigned int m_currentIteration;
	double m_previousModeFitness;
	double m_previousIterationFitness;
	unsigned int m_frozenOnesSize;

	platform::Percent m_percentOfSupportVectorsThreshold;
	const unsigned m_iterationsBeforeChangeThreshold;
	const unsigned int m_maxNumberOfClassExamples;

	bool m_isLocalModeInitialValue;
};

class CompensationInformationRbfLinear
{
public:
	explicit CompensationInformationRbfLinear(std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
	                                          unsigned int numberOfClasses);

	std::vector<unsigned int> generate(const std::vector<geneticComponents::Parents<SvmCustomKernelChromosome>>& parents,
	                                   unsigned int numberOfClassExamples) const;

private:
	std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	const unsigned int m_numberOfClasses;
};

class MutationRbfLinear : public IMutationOperator<SvmCustomKernelChromosome>
{
public:
	explicit MutationRbfLinear(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
	                           platform::Percent exchangePercent,
	                           platform::Percent mutationProbability,
	                           const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                           const std::vector<unsigned int>& labelsCount);

	void mutatePopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population) override;
	void mutateChromosome(SvmCustomKernelChromosome& chromosome) override;

	void setGamma(double gamma)
	{
		m_gamma = gamma;
	}

	void setTrainingSet(dataset::Dataset<std::vector<float>, float>& trainingSet)
	{
		m_trainingSet = trainingSet;
	}

	void setForbbidens(const std::unordered_set<uint64_t>& ids)
	{
		m_forbiddenIds = ids;
	}

private:
	std::unordered_set<uint64_t> setDifference(const std::vector<Gene>& set, const std::unordered_set<uint64_t>& deleted);

	void calculateNumberOfPossibleExchanges(SvmCustomKernelChromosome& chromosome,
	                                        std::vector<uint64_t>& possibleNumberOfExchangesPerClass,
	                                        const std::vector<uint64_t>& forbiddenExchangesNumber) const;

	std::vector<Gene> findReplacement(const std::unordered_set<uint64_t>& deleted,
	                                  std::unordered_set<uint64_t>& mutated,
	                                  const std::vector<std::size_t>& positionsToReplace,
	                                  SvmCustomKernelChromosome& chromosome) const;

	void getPositionsOfMutation(SvmCustomKernelChromosome& chromosome,
	                            std::unordered_set<uint64_t>& deleted,
	                            std::vector<std::size_t>& positionsToReplace) const;

	std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
	platform::Percent m_mutationProbability;
	platform::Percent m_exchangePercent;
	unsigned int m_numberOfExchanges;
	const unsigned int m_numberOfClasses;
	const std::vector<unsigned int> m_labelsCount;
	dataset::Dataset<std::vector<float>, float> m_trainingSet;

	double m_gamma;
	std::unordered_set<uint64_t> m_forbiddenIds;
};
} // namespace svmComponents
