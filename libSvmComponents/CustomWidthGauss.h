#pragma once

#include "LibGeneticComponents/IPopulationGeneration.h"
#include "SvmCustomKernelChromosome.h"
#include "libRandom/IRandomNumberGenerator.h"

namespace svmComponents
{
static std::vector<double> GAMMAS_GLOBAL = { 10,1000 };

class CusomKernelGeneration : public geneticComponents::IPopulationGeneration<SvmCustomKernelChromosome>
{
public:
	explicit CusomKernelGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                               std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
	                               unsigned int numberOfClassExamples,
	                               const std::vector<unsigned int>& labelsCount);

	geneticComponents::Population<SvmCustomKernelChromosome> createPopulation(uint32_t populationSize) override;

	void setCandGamma(std::vector<double>& C, std::vector<double>& gamma)
	{
		m_C = C;
		m_gamma = gamma;
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

private:
	dataset::Dataset<std::vector<float>, float> m_trainingSet;
	std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
	unsigned int m_numberOfClassExamples;
	unsigned int m_numberOfClasses;
	std::vector<double> m_C;
	std::vector<double> m_gamma;
    double m_gammaSingle;
};

inline CusomKernelGeneration::CusomKernelGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                    std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                                    unsigned int numberOfClassExamples,
                                                    const std::vector<unsigned int>& labelsCount)
	: m_trainingSet(trainingSet)
	, m_rngEngine(std::move(rngEngine))
	, m_numberOfClassExamples(numberOfClassExamples) //*std::min_element(labelsCount.begin(), labelsCount.end())
	, m_numberOfClasses(static_cast<unsigned int>(labelsCount.size()))
{
	if (std::any_of(labelsCount.begin(), labelsCount.end(), [this](const auto& labelCount)
	{
		return m_numberOfClassExamples > labelCount;
	}))
	{
		//throw ValueOfClassExamplesIsTooHighForDataset(m_numberOfClassExamples);
	}
    m_gammaSingle = 10;
}

inline geneticComponents::Population<SvmCustomKernelChromosome> CusomKernelGeneration::createPopulation(uint32_t populationSize)
{
	//populationSize = 20;
	if (populationSize == 0)
	{
		throw geneticComponents::PopulationIsEmptyException();
	}

	auto samples = m_trainingSet.getSamples();
	auto targets = m_trainingSet.getLabels();
	auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));
	std::vector<SvmCustomKernelChromosome> population(populationSize);

	//// Code for using the same training set for all individuals in population
	//std::unordered_set<std::uint64_t> trainingSetOnce;
	//std::vector<unsigned int> classCount(m_numberOfClasses, 0);
	//while (std::any_of(classCount.begin(), classCount.end(), [this](const auto& classIndicies)
	//{
	//	return classIndicies != m_numberOfClassExamples;
	//}))
	//{
	//	auto randomValue = m_rngEngine->getRandom(trainingSetID);
	//	if (classCount[static_cast<int>(targets[randomValue])] < m_numberOfClassExamples && // less that desired number of class examples
	//		trainingSetOnce.emplace(static_cast<int>(randomValue)).second) // is unique
	//	{
	//		classCount[static_cast<int>(targets[randomValue])]++;
	//	}
	//}

	std::generate(population.begin(), population.end(), [&]
	{
		//auto it = trainingSetOnce.begin();
		std::unordered_set<std::uint64_t> trainingSet;
		std::vector<Gene> chromosomeDataset;
		chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
		std::vector<unsigned int> classCount(m_numberOfClasses, 0);
		while (std::any_of(classCount.begin(), classCount.end(), [this](const auto& classIndicies)
		{
			return classIndicies != m_numberOfClassExamples;
		}))
		{
			//auto gammaValueRandom = std::uniform_real_distribution<double>(*std::min_element(m_gamma.begin(), m_gamma.end()), *std::max_element(m_gamma.begin(), m_gamma.end()));

			//gammas 10 100 1000
			std::vector<double> gammas = GAMMAS_GLOBAL;
			auto gammaValueRandom = std::uniform_int_distribution<int>(0, static_cast<int>(gammas.size())-1);

			auto randomValue = m_rngEngine->getRandom(trainingSetID); //*it;
			//++it;
			if (classCount[static_cast<int>(targets[randomValue])] < m_numberOfClassExamples && // less that desired number of class examples
				trainingSet.emplace(static_cast<int>(randomValue)).second) // is unique
			{
				auto gammaValue = gammas[m_rngEngine->getRandom(gammaValueRandom)]; //m_rngEngine->getRandom(gammaValueRandom);
				chromosomeDataset.emplace_back(Gene(randomValue, static_cast<std::uint8_t>(targets[randomValue]), gammaValue));	           
				classCount[static_cast<int>(targets[randomValue])]++;
			}
		}

		//auto cValueRandom = std::uniform_real_distribution<double>(*std::min_element(m_C.begin(), m_C.end()), *std::max_element(m_C.begin(), m_C.end()));
		return SvmCustomKernelChromosome(std::move(chromosomeDataset), 1);
	});




	return geneticComponents::Population<SvmCustomKernelChromosome>(std::move(population));
}


class MutationCustomGauss : public geneticComponents::IMutationOperator<SvmCustomKernelChromosome>
{
public:
	explicit MutationCustomGauss(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
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

private:
	std::unordered_set<uint64_t> setDifference(const std::vector<Gene>& set, const std::unordered_set<uint64_t>& deleted);

	void calculateNumberOfPossibleExchanges(SvmCustomKernelChromosome& chromosome,
	                                        std::vector<uint64_t>& possibleNumberOfExchangesPerClass) const;

	std::vector<Gene> findReplacement(const std::unordered_set<uint64_t>& deleted,
	                                  std::unordered_set<uint64_t>& mutated,
	                                  const std::vector<std::size_t>& positionsToReplace,
	                                  SvmCustomKernelChromosome& chromosome) const;

	void getPositionsOfMutation(SvmCustomKernelChromosome& chromosome,
	                            std::unordered_set<uint64_t>& deleted,
	                            std::vector<std::size_t>& positionsToReplace) const;

	std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
	platform::Percent m_mutationProbability;	
	platform::Percent m_exchangePercent;
	unsigned int m_numberOfExchanges;
	const unsigned int m_numberOfClasses;
	const std::vector<unsigned int> m_labelsCount;
	dataset::Dataset<std::vector<float>, float> m_trainingSet;

	double m_gamma;

};

inline MutationCustomGauss::MutationCustomGauss(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                         platform::Percent exchangePercent,
                                         platform::Percent mutationProbability,
                                         const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                         const std::vector<unsigned int>& labelsCount)
	: m_rngEngine(std::move(rngEngine))
	, m_mutationProbability(mutationProbability)
	, m_exchangePercent(exchangePercent)
	, m_numberOfExchanges(0)
	, m_numberOfClasses(static_cast<unsigned int>(labelsCount.size()))
	, m_labelsCount(labelsCount)
	, m_trainingSet(trainingSet)
{
	m_gamma = 10;
}

inline void MutationCustomGauss::mutatePopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population)
{
	if (population.empty())
	{
		throw geneticComponents::PopulationIsEmptyException();
	}

	std::bernoulli_distribution mutation(m_mutationProbability.m_percentValue); //range 0% to 100%
	for (auto& chromosome : population)
	{
		if (m_rngEngine->getRandom(mutation))
		{
			mutateChromosome(chromosome);
		}
	}
}

inline std::unordered_set<uint64_t> MutationCustomGauss::setDifference(const std::vector<Gene>& set,
	const std::unordered_set<uint64_t>& deleted)
{
	std::vector<Gene> temporary;
	std::copy_if(set.begin(),
		set.end(),
		std::back_inserter(temporary),
		[&deleted](const auto& dataVector)
	{
		return deleted.find(dataVector.id) == deleted.end();
	});

	std::unordered_set<uint64_t> difference;
	difference.reserve(set.size());
	for (const auto& dataVector : temporary)
	{
		difference.insert(dataVector.id);
	}
	return difference;
}

inline void MutationCustomGauss::calculateNumberOfPossibleExchanges(SvmCustomKernelChromosome& chromosome,
                                                                    std::vector<uint64_t>& possibleNumberOfExchangesPerClass) const
{
	for (auto i = 0u; i < m_numberOfClasses; ++i)
	{
		possibleNumberOfExchangesPerClass[i] = m_labelsCount[i] - (chromosome.size() / m_numberOfClasses);
	}
}

inline std::vector<Gene> MutationCustomGauss::findReplacement(const std::unordered_set<uint64_t>& deleted,
                                                                       std::unordered_set<uint64_t>& mutated,
                                                                       const std::vector<std::size_t>& positionsToReplace,
                                                                       SvmCustomKernelChromosome& chromosome) const
{
	std::uniform_int_distribution<int> datasetPosition(0, static_cast<int>(m_trainingSet.size() - 1));
	const auto targets = m_trainingSet.getLabels();
	auto dataset = chromosome.getDataset();

	std::vector<uint64_t> possibleNumberOfExchangesPerClass(m_numberOfClasses);
	calculateNumberOfPossibleExchanges(chromosome, possibleNumberOfExchangesPerClass);

	//gammas
	//std::vector<double> gammas = { m_gamma,m_gamma };
	std::vector<double> gammas = { m_gamma }; // GAMMAS_GLOBAL;//{ 10,50,100,250,500,750,1000 };
	auto gammaValueRandom = std::uniform_int_distribution<int>(0, static_cast<int>(gammas.size()) - 1);


	for (auto i = 0u; i < m_numberOfExchanges; i++)
	{
		while (true)
		{
			auto newId = m_rngEngine->getRandom(datasetPosition);
			if (static_cast<int>(dataset[positionsToReplace[i]].classValue) == targets[newId] && // @wdudzik class value match
				deleted.find(newId) == deleted.end() &&
				mutated.emplace(newId).second) // @wdudzik is newId unique in chromosome dataset
			{
				dataset[positionsToReplace[i]].id = newId;
				dataset[positionsToReplace[i]].gamma = gammas[m_rngEngine->getRandom(gammaValueRandom)];
				possibleNumberOfExchangesPerClass[static_cast<int>(dataset[positionsToReplace[i]].classValue)]--;
				break;
			}
			if (possibleNumberOfExchangesPerClass[static_cast<int>(dataset[positionsToReplace[i]].classValue)] == 0)
			{
				break;
			}
		}
	}
	return dataset;
}

inline void MutationCustomGauss::getPositionsOfMutation(SvmCustomKernelChromosome& chromosome,
	std::unordered_set<uint64_t>& deleted,
	std::vector<std::size_t>& positionsToReplace) const
{
	auto& dataset = chromosome.getDataset();
	std::uniform_int_distribution<int> replacePosition(0, static_cast<int>(dataset.size() - 1));
	for (auto i = 0u; i < m_numberOfExchanges;)
	{
		auto position = m_rngEngine->getRandom(replacePosition);
		if (deleted.insert(dataset[position].id).second) // @wdudzik if position is unique
		{
			positionsToReplace.emplace_back(position);
			++i;
		}
	}
}

inline void MutationCustomGauss::mutateChromosome(SvmCustomKernelChromosome& chromosome)
{
	std::unordered_set<uint64_t> deleted;
	std::vector<std::size_t> positionsToReplace;

	m_numberOfExchanges = static_cast<unsigned int>(std::floor(chromosome.size() * m_exchangePercent.m_percentValue));
	positionsToReplace.reserve(m_numberOfExchanges);
	deleted.reserve(m_numberOfExchanges);

	// @wdudzik get values to be exchanged (mutated), and save into deleted
	getPositionsOfMutation(chromosome, deleted, positionsToReplace);

	//@wdudzik set difference between chromosome dataset and deleted
	auto mutated = setDifference(chromosome.getDataset(), deleted);

	//@wdudzik find and insert new ones. Restrictions are: no duplicates, cannot insert what was deleted, number of class examples have to match
	auto mutatedDataset = findReplacement(deleted, mutated, positionsToReplace, chromosome);
	chromosome.updateDataset(mutatedDataset);
}



//inline MutationCustomGauss::MutationCustomGauss(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
//                                         platform::Percent maxPercentChange,
//                                         platform::Percent mutationProbability)
//	: m_rngEngine(std::move(rngEngine))
//	, m_mutationProbability(mutationProbability)
//	, m_maxPercentChange(maxPercentChange)
//{
//}
//
//inline void MutationCustomGauss::mutatePopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population)
//{
//	if (population.empty())
//	{
//		throw geneticComponents::PopulationIsEmptyException();
//	}
//
//	std::bernoulli_distribution mutation(m_mutationProbability.m_percentValue);
//	std::sort(population.begin(), population.end());
//
//	for (auto& chromosome : population)
//	{
//		if (m_rngEngine->getRandom(mutation))
//		{
//			mutateChromosome(chromosome);
//		}
//	}
//}
//
//inline void MutationCustomGauss::mutateChromosome(SvmCustomKernelChromosome& chromosome)
//{
//	std::uniform_real_distribution<double> mutation(platform::Percent::m_minPercent, m_maxPercentChange.m_percentValue);
//
//	auto newParameters(chromosome.getDataset());
//	auto oldParameters = chromosome.getDataset();
//
//	std::transform(oldParameters.begin(),
//	               oldParameters.end(),
//	               newParameters.begin(),
//	               newParameters.begin(),
//	               [&, this](Gene& oldParameter, Gene& newParameter)
//	               {
//		               auto sign = getRandomSign();
//		               auto newValue = m_rngEngine->getRandom(mutation) * (oldParameter.gamma * sign) + newParameter.gamma;
//		               if (newValue <= 0)
//		               {
//			               return oldParameter;
//		               }
//		               return Gene(oldParameter.id, oldParameter.classValue, newValue);
//	               });
//	chromosome.updateDataset(newParameters);
//}
//
//inline int MutationCustomGauss::getRandomSign() const
//{
//	std::uniform_real_distribution<double> sign(platform::Percent::m_minPercent, platform::Percent::m_maxPercent);
//	constexpr double halfRange = platform::Percent::m_maxPercent / 2;
//	constexpr auto positive = 1;
//	constexpr auto negative = -1;
//	return m_rngEngine->getRandom(sign) > halfRange ? positive : negative;
//}

class CrossoverCustomGauss : public geneticComponents::BaseCrossoverOperator<SvmCustomKernelChromosome>
{
public:
	using chromosomeType = SvmCustomKernelChromosome;

	explicit CrossoverCustomGauss(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
	                              unsigned int numberOfClasses);

	chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB) override;

	void setImbalancedOrOneClass(bool value)
	{
		m_imbalancedOrOneClass = value;
	}

private:
	void tryToInsert(const std::vector<Gene>::const_iterator& parentAIt,
	                 const std::vector<Gene>::const_iterator& parentBIt,
	                 std::unordered_set<uint64_t>& childSet,
	                 std::vector<unsigned>& classes,
	                 unsigned int classExamples,
	                 std::vector<Gene>& child) const;
	std::vector<Gene> crossoverInternals(const chromosomeType& parentA,
	                                     const chromosomeType& parentB,
	                                     unsigned int datasetSize,
	                                     unsigned int classExamples);

	std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
	unsigned int m_numberOfClasses;
	bool m_imbalancedOrOneClass;
};

inline CrossoverCustomGauss::CrossoverCustomGauss(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine, unsigned numberOfClasses)
	: m_rngEngine(std::move(rngEngine))
	, m_numberOfClasses(numberOfClasses)
	, m_imbalancedOrOneClass(false)
{
	if (m_rngEngine == nullptr)
	{
		throw RandomNumberGeneratorNullPointer();
	}
}

inline CrossoverCustomGauss::chromosomeType CrossoverCustomGauss::crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB)
{
	auto sizeOfChild = std::max(parentA.getDataset().size(), parentB.getDataset().size());
	auto datasetSize = sizeOfChild;
	auto classExamples = sizeOfChild / m_numberOfClasses;

	std::vector<Gene> child;
	
	if(parentA.getDataset().size() == 0) //special cases when super individuals are empty (but there are vectors in frozen set)
	{
		child = parentA.getDataset();
	}
	else if (parentB.getDataset().size() == 0)
	{
		child = parentB.getDataset();
	}
	else
	{
		child = crossoverInternals(parentA, parentB, static_cast<unsigned int>(datasetSize), static_cast<unsigned int>(classExamples));
	}

	//C hiperparameter crossover
	//auto alpha = m_rngEngine->getRandom(std::uniform_real_distribution<double>(0.5,1.5)); //Magic numbers (same as default in other GA)
	//auto minMaxPair = std::minmax(parentA, parentB);
	//const auto& lowFitnessParent = minMaxPair.first;
	//const auto& highFitnessParent = minMaxPair.second;
	//auto newC = lowFitnessParent.getC() + alpha * (highFitnessParent.getC() - lowFitnessParent.getC());
	//
	//if(newC < 0)
	//{
	//	newC = highFitnessParent.getC();
	//}

	SvmCustomKernelChromosome ch(std::move(child), parentA.getC());  //TODO: assumption that both elements have the same C hiperparameter value
	ch.updateClassifier(parentA.getClassifier()); //Only to pass information about the kernel it will get overwrite on training
	return ch;
}

inline void CrossoverCustomGauss::tryToInsert(const std::vector<Gene>::const_iterator& parentAIt, const std::vector<Gene>::const_iterator& parentBIt,
                                              std::unordered_set<uint64_t>& childSet, std::vector<unsigned>& classes, unsigned classExamples,
                                              std::vector<Gene>& child) const
{
	//imbalanced case
	if(m_imbalancedOrOneClass)
	{
		if (childSet.insert(parentAIt->id).second)
		{
			++classes[0];
			child.emplace_back(*parentAIt);
		}
		else if ( childSet.insert(parentBIt->id).second)
		{
			++classes[0];
			child.emplace_back(*parentBIt);
		}
		return;
	}

	if (classes[static_cast<int>(parentAIt->classValue)] < classExamples && childSet.insert(parentAIt->id).second)
	{
		++classes[static_cast<int>(parentAIt->classValue)];
		child.emplace_back(*parentAIt);
	}
	else if (classes[static_cast<int>(parentBIt->classValue)] < classExamples && childSet.insert(parentBIt->id).second)
	{
		++classes[static_cast<int>(parentBIt->classValue)];
		child.emplace_back(*parentBIt);
	}
}

inline std::vector<Gene> CrossoverCustomGauss::crossoverInternals(const chromosomeType& parentA,
                                                                  const chromosomeType& parentB,
                                                                  unsigned datasetSize,
                                                                  unsigned classExamples)
{
	std::vector<Gene> child;
	std::vector<unsigned int> classes;
	std::unordered_set<uint64_t> childSet;
	child.reserve(datasetSize);
	childSet.reserve(datasetSize);
	classes.resize(m_numberOfClasses);

	auto parentAIt = parentA.getDataset().begin();
	auto parentBIt = parentB.getDataset().begin();
	int finalBreak = 0;
	while (true && finalBreak < 100000)
	{
		std::uniform_real_distribution<double> randomParent(platform::Percent::m_minPercent, platform::Percent::m_maxPercent);
		constexpr auto halfRange = platform::Percent::m_maxPercent / 2;
		auto parentChoose = m_rngEngine->getRandom(randomParent);
		if (parentChoose > halfRange)
		{
			tryToInsert(parentAIt, parentBIt, childSet, classes, classExamples, child);
		}
		else
		{
			tryToInsert(parentBIt, parentAIt, childSet, classes, classExamples, child);
		}
		if(m_imbalancedOrOneClass)
		{
			if (std::any_of(classes.begin(), classes.end(), [&](const auto classCount)
				{
					return classCount == m_numberOfClasses * classExamples;
				}))
			{
				break;
			}
		}
		else
		{
			if (std::all_of(classes.begin(), classes.end(), [classExamples](const auto classCount)
				{
					return classCount == classExamples;
				}))
			{
				break;
			}
		}
		if (++parentBIt == parentB.getDataset().end())
		{
			parentBIt = parentB.getDataset().begin();
		}
		if (++parentAIt == parentA.getDataset().end())
		{
			parentAIt = parentA.getDataset().begin();
		}
		finalBreak++;
	}
	return child;
}
} // namespace svmComponents
