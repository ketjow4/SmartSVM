
#include <opencv2/ml.hpp>
#include "SvmLib/SvmExceptions.h"
#include "MultipleGammaInternals.h"
#include "SvmLib/SvmLibImplementation.h"

namespace svmComponents
{
MultipleGammaGeneration::MultipleGammaGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
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
		//m_gammaSingle = 0;
		m_forbiddenIds = {};
		m_imbalancedOrOneClass = false;
}

geneticComponents::Population<SvmCustomKernelChromosome> MultipleGammaGeneration::createPopulation(uint32_t populationSize)
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

	std::generate(population.begin(), population.end(), [&]
		{
			//auto it = trainingSetOnce.begin();
			std::unordered_set<std::uint64_t> trainingSet;
			std::vector<Gene> chromosomeDataset;
			chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
			std::vector<unsigned int> classCount(m_numberOfClasses, 0);
			while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies)
				{
					return m_imbalancedOrOneClass ? classCount[0] != m_numberOfClassExamples : classIndicies != m_numberOfClassExamples;
				}))
			{
				auto gammaValueRandom = std::uniform_int_distribution<int>(0, static_cast<int>(m_gamma.size() - 1));
				//std::vector<double> gammas = { m_gammaSingle };
				//auto gammaValueRandom = std::uniform_int_distribution<int>(0, static_cast<int>(gammas.size()) - 1);

				auto randomValue = m_rngEngine->getRandom(trainingSetID); //*it;
				//++it;
				if ((!m_imbalancedOrOneClass
					&& classCount[static_cast<int>(targets[randomValue])] < m_numberOfClassExamples  // less that desired number of class examples
					&& trainingSet.emplace(static_cast<int>(randomValue)).second  // is unique
					&& (m_forbiddenIds.empty() || m_forbiddenIds.find(randomValue) == m_forbiddenIds.end()) // not in forbidden
					)
					||
					(m_imbalancedOrOneClass
						&& classCount[0] < m_numberOfClassExamples
						&& trainingSet.emplace(static_cast<int>(randomValue)).second  // is unique)
						&& (m_forbiddenIds.empty() || m_forbiddenIds.find(randomValue) == m_forbiddenIds.end()))
					)
				{
					auto gammaValue = m_gamma[m_rngEngine->getRandom(gammaValueRandom)]; 
					chromosomeDataset.emplace_back(Gene(randomValue, static_cast<std::uint8_t>(targets[randomValue]), gammaValue));
					if (m_imbalancedOrOneClass)
					{
						classCount[0]++;
					}
					else
					{
						classCount[static_cast<int>(targets[randomValue])]++;
					}
					if (m_trainingSet.size() - m_forbiddenIds.size() == chromosomeDataset.size())
						break;

				}
			}

				//auto cValueRandom = std::uniform_real_distribution<double>(*std::min_element(m_C.begin(), m_C.end()), *std::max_element(m_C.begin(), m_C.end()));
				return SvmCustomKernelChromosome(std::move(chromosomeDataset), m_CSingle);
		});

	return geneticComponents::Population<SvmCustomKernelChromosome>(std::move(population));
}








MultipleGammaCrossoverCompensation::MultipleGammaCrossoverCompensation(const dataset::Dataset<std::vector<float>, float>& trainingSet,
		std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
		unsigned int numberOfClasses)
		: m_trainingSet(trainingSet)
		, m_rngEngine(std::move(rngEngine))
		, m_numberOfClasses(numberOfClasses)
		//, m_gamma(0)
	{
		m_forbiddenIds = {};
		m_imbalancedOrOneClass = false;
	}

geneticComponents::Population<SvmCustomKernelChromosome> MultipleGammaCrossoverCompensation::compensate(geneticComponents::Population<SvmCustomKernelChromosome>& population,
	const std::vector<unsigned int>& compensationInfo)
{
	if (compensationInfo.size() == population.size())
	{
		auto index = 0u;
		const auto targets = m_trainingSet.getLabels();
		const auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));

		for (auto& individual : population)
		{
			std::vector<unsigned int> classCount(m_numberOfClasses, static_cast<const unsigned>(individual.getDataset().size() / m_numberOfClasses));
			auto numberOfClassExamples = (individual.getDataset().size() + compensationInfo[index]) / m_numberOfClasses;
			auto dataset = individual.getDataset();
			auto trainingSet = individual.convertToSet();
			int j = 0;
			auto gammaValueRandom = std::uniform_int_distribution<int>(0, static_cast<int>(m_gamma.size() - 1));
			while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies)
				{
					return m_imbalancedOrOneClass ? classCount[0] != numberOfClassExamples : classIndicies != numberOfClassExamples;
				}))
			{
				auto randomValue = m_rngEngine->getRandom(trainingSetID);
				if ((!m_imbalancedOrOneClass
					&& classCount[static_cast<int>(targets[randomValue])] < numberOfClassExamples  // less that desired number of class examples
					&& trainingSet.emplace(static_cast<int>(randomValue)).second  // is unique
					&& (m_forbiddenIds.empty() || m_forbiddenIds.find(randomValue) == m_forbiddenIds.end()) // not in forbidden
					)
					||
					(m_imbalancedOrOneClass
						&& classCount[0] < numberOfClassExamples
						&& trainingSet.emplace(static_cast<int>(randomValue)).second  // is unique)
						&& (m_forbiddenIds.empty() || m_forbiddenIds.find(randomValue) == m_forbiddenIds.end()))
					)
				{
					auto gammaValue = m_gamma[m_rngEngine->getRandom(gammaValueRandom)];
					dataset.emplace_back(Gene(randomValue, static_cast<std::uint8_t>(targets[randomValue]), gammaValue));
					if (m_imbalancedOrOneClass)
					{
						classCount[0]++;
					}
					else
					{
						classCount[static_cast<int>(targets[randomValue])]++;
					}
					//++classCount[static_cast<int>(targets[randomValue])];
				}

				j++;
				if (j > 100000) //FIX THIS IN FUTURE
					break;
			}
			individual.updateDataset(dataset);
			index++;
		}
	}
	else
	{
		//throw ContainersSizeInequality("libSvmComponents CrossoverCompensation::compensate", compensationInfo.size(), population.size());
	}
	return population;
}






MultipleGammaEducationOfTrainingSet::MultipleGammaEducationOfTrainingSet(platform::Percent educationProbability,
		unsigned int numberOfClasses,
		std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
		std::unique_ptr<ISupportVectorSelectionGamma> supportVectorSelection)
		: m_educationProbability(educationProbability)
		, m_numberOfClasses(numberOfClasses)
		, m_rngEngine(std::move(randomNumberGenerator))
		, m_supportVectorSelection(std::move(supportVectorSelection))
	{
		if (m_rngEngine == nullptr)
		{
			//throw RandomNumberGeneratorNullPointer();
		}
		if (m_supportVectorSelection == nullptr)
		{
			//throw MemberNullPointer("m_supportVectorSelection");
		}
	}

void MultipleGammaEducationOfTrainingSet::educatePopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population,
	const std::vector<Gene>& supportVectorPool,
	const std::vector<geneticComponents::Parents<SvmCustomKernelChromosome>>& parents,
	const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
	auto parentIterator = parents.begin();
	for (auto& individual : population)
	{
		educate(individual, *parentIterator, supportVectorPool, trainingSet);
		++parentIterator;
	}
}

bool MultipleGammaEducationOfTrainingSet::replacementCondition(const Gene& supportVectorPoolElement,
	std::unordered_set<std::uint64_t>& traningIDs,
	const Gene& sample)
{
	return sample.classValue == supportVectorPoolElement.classValue && // @wdudzik class value match
		traningIDs.emplace(supportVectorPoolElement.id).second; // @wdudzik is id of supportVector unique in chromosome dataset
}

void MultipleGammaEducationOfTrainingSet::educate(SvmCustomKernelChromosome& individual,
	const geneticComponents::Parents<SvmCustomKernelChromosome>& parents,
	const std::vector<Gene>& supportVectorPool,
	const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
	auto dataset = individual.getDataset();
	auto parentsSupportVectors = findSupportVectors(parents, trainingSet);
	auto weakSamples = setDifference(parentsSupportVectors, dataset);
	auto traningIDs = individual.convertToSet();

	//@wdudzik SupportVectorPool - traningIDs; possible replecement for weakSamples
	auto result = setDifference(traningIDs, supportVectorPool);

	if (result.empty())
	{
		return;
	}

	auto numberOfPossibleExchanges = svmUtils::countLabels(m_numberOfClasses, result);
	std::bernoulli_distribution education(m_educationProbability.m_percentValue);
	const auto endIndex = static_cast<int>(supportVectorPool.size() - 1);
	std::uniform_int_distribution<int> supportVectorPoolID(0, endIndex);

	for (const auto& sample : weakSamples)
	{
		if (m_rngEngine->getRandom(education))
		{
			auto positionInDataset = std::find(dataset.begin(), dataset.end(), sample);

			//@wdudzik try to replace with one form supportVectorPool
			int stop_inifinite = 0;
			while (true && stop_inifinite < 100000)
			{
				auto newId = m_rngEngine->getRandom(supportVectorPoolID);
				if (replacementCondition(supportVectorPool[newId], traningIDs, sample))
				{
					numberOfPossibleExchanges[static_cast<int>(sample.classValue)]--;
					*positionInDataset = supportVectorPool[newId];
					break;
				}
				if (numberOfPossibleExchanges[static_cast<int>(sample.classValue)] == 0)
				{
					break;
				}
				stop_inifinite++;
			}
		}
	}
	individual.updateDataset(dataset);
}

std::vector<Gene> MultipleGammaEducationOfTrainingSet::setDifference(const std::unordered_set<uint64_t>& svPool,
	const std::vector<Gene>& traningDataset)
{
	std::vector<Gene> weakSamples;
	std::copy_if(traningDataset.begin(),
		traningDataset.end(),
		std::back_inserter(weakSamples),
		[&svPool](const auto& dataVector)
		{
			return svPool.find(dataVector.id) == svPool.end();
		});
	return weakSamples;
}

std::unordered_set<uint64_t> MultipleGammaEducationOfTrainingSet::findSupportVectors(const geneticComponents::Parents<SvmCustomKernelChromosome>& parents,
	const dataset::Dataset<std::vector<float>, float>& trainingSet) const
{
	m_supportVectorSelection->addSupportVectors(parents.first, trainingSet);
	m_supportVectorSelection->addSupportVectors(parents.second, trainingSet);

	return m_supportVectorSelection->getSupportVectorIds();
}










const std::vector<Gene>& MultipleGammaSupportVectorPool::getSupportVectorPool() const
{
	return m_supportVectorPool;
}

const std::unordered_set<uint64_t>& MultipleGammaSupportVectorPool::getSupportVectorIds() const
{
	return m_supportVectorIds;
}

void MultipleGammaSupportVectorPool::updateSupportVectorPool(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
	const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	for (const auto& individual : population)
	{
		if(individual.getFitness() >= population.getMeanFitness())
		{
			addSupportVectors(individual, trainingSet);
		}
	}
}

unsigned int MultipleGammaSupportVectorPool::findPositionOfSupprotVector(const dataset::Dataset<std::vector<float>, float>& individualDataset,
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

void MultipleGammaSupportVectorPool::addSupportVectors(const SvmCustomKernelChromosome& chromosome,
	const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	const auto classifier = chromosome.getClassifier();
	if (classifier && classifier->isTrained())
	{
		//auto svm = reinterpret_cast<phd::svm::SvmLibImplementation*>(classifier.get());
		//auto[ommit, scores] = svm->check_sv(trainingSet);
		cv::Mat supportVectors = classifier->getSupportVectors();
		auto individualDataset = chromosome.convertChromosome(trainingSet);

		/*if (!classifier->getFeatureSet().empty())
		{
			auto featureSet = classifier->getFeatureSet();
			SvmFeatureSetMemeticChromosome c(std::move(featureSet));
			individualDataset = c.convertChromosome(individualDataset);
		}*/

		auto& dataset = chromosome.getDataset();

		for (auto i = 0; i < supportVectors.rows; i++)
		{
			//if (scores[i] > 0.95)
			{
				const float* sv = supportVectors.ptr<float>(i);
				const gsl::span<const float> supportVector(sv, supportVectors.cols);

				const auto positionInDataset = findPositionOfSupprotVector(individualDataset, supportVector);

				if (positionInDataset == individualDataset.size())
					continue; //the case of SV that is not in the individual dataset (so one of the frozen ones)

				if(m_supportVectorIds.emplace(dataset[positionInDataset].id).second)
				{
					m_supportVectorPool.emplace_back(dataset[positionInDataset].id, dataset[positionInDataset].classValue, dataset[positionInDataset].gamma);
				}
				else
				{
					auto duplicate = std::find_if(m_supportVectorPool.begin(), m_supportVectorPool.end(), [&](auto& element)
					{
							return element.id == dataset[positionInDataset].id && element.gamma == dataset[positionInDataset].gamma;
						});
					if(duplicate == m_supportVectorPool.end())
					{
						m_supportVectorPool.emplace_back(dataset[positionInDataset].id, dataset[positionInDataset].classValue, dataset[positionInDataset].gamma);
					}
				}
			}
		}
		return;
	}
	throw phd::svm::UntrainedSvmClassifierException();
}







using namespace geneticComponents;

MultipleGammaSuperIndividualsCreation::MultipleGammaSuperIndividualsCreation(std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
	unsigned int numberOfClasses)
	: m_rngEngine(std::move(randomNumberGenerator))
	, m_numberOfClasses(numberOfClasses)
	, m_imbalancedOrOneClass(false)
	, m_Cvalue(1.0)
{
	if (m_rngEngine == nullptr)
	{
		throw RandomNumberGeneratorNullPointer();
	}
}

bool emplacementCondition2(const Gene& drawnSupportVector,
	const std::vector<unsigned int>& classCount,
	const unsigned int numberOfClassExamples,
	std::unordered_set<std::uint64_t>& trainingSet,
	bool m_imbalancedOrOneClass)
{
	if (m_imbalancedOrOneClass)
	{
		return  classCount[0] < numberOfClassExamples && // less that desired number of class examples
			trainingSet.emplace(static_cast<int>(drawnSupportVector.id)).second; // is unique
	}
	else
	{
		return classCount[static_cast<int>(drawnSupportVector.classValue)] < numberOfClassExamples && // less that desired number of class examples
			trainingSet.emplace(static_cast<int>(drawnSupportVector.id)).second; // is unique
	}
}

std::vector<SvmCustomKernelChromosome> MultipleGammaSuperIndividualsCreation::generate(unsigned int populationSize,
	const std::vector<Gene>& supportVectorPool,
	unsigned int numberOfClassExamples)
{
	std::vector<SvmCustomKernelChromosome> population(populationSize);
	auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(supportVectorPool.size() - 1));

	std::generate(population.begin(), population.end(), [&]
		{
			std::unordered_set<std::uint64_t> trainingSet;
			std::vector<Gene> superIndividualDataset;
			superIndividualDataset.reserve(numberOfClassExamples * m_numberOfClasses);
			std::vector<unsigned int> classCount(m_numberOfClasses, 0);
			int j = 0;
			while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies)
				{
					return m_imbalancedOrOneClass ? classCount[0] != numberOfClassExamples : classIndicies != numberOfClassExamples;
				}) && j < 10000)
			{
				auto randomValue = m_rngEngine->getRandom(trainingSetID);
				const auto& drawnSupportVector = supportVectorPool[randomValue];
				if (emplacementCondition2(drawnSupportVector, classCount, numberOfClassExamples, trainingSet, m_imbalancedOrOneClass))
				{
					superIndividualDataset.emplace_back(Gene(drawnSupportVector));
					if (m_imbalancedOrOneClass)
					{
						classCount[0]++;
					}
					else
					{
						classCount[static_cast<int>(drawnSupportVector.classValue)]++;
					}
				}
				j++;
				if (trainingSet.size() == supportVectorPool.size())
					break;
			}
				return SvmCustomKernelChromosome(std::move(superIndividualDataset), m_Cvalue);
		});
	return population;
}

Population<SvmCustomKernelChromosome> MultipleGammaSuperIndividualsCreation::createPopulation(uint32_t populationSize,
	const std::vector<Gene>& supportVectorPool,
	unsigned int numberOfClassExamples)
{
	if (populationSize == 0)
	{
		throw PopulationIsEmptyException();
	}
	if (supportVectorPool.empty())
	{
		throw EmptySupportVectorPool();
	}

	auto labelsCount = svmUtils::countLabels(m_numberOfClasses, supportVectorPool);
	auto minLabelCount = std::min_element(labelsCount.begin(), labelsCount.end());

	if (numberOfClassExamples > *minLabelCount && !m_imbalancedOrOneClass)
	{
		numberOfClassExamples = *minLabelCount;
	}

	auto population = generate(populationSize, supportVectorPool, numberOfClassExamples);

	return Population<SvmCustomKernelChromosome>(std::move(population));
}




	bool MultipleGammaMemeticTrainingSetAdaptation::getIsModeLocal() const
	{
		return m_isLocalMode;
	}

	unsigned int MultipleGammaMemeticTrainingSetAdaptation::getNumberOfClassExamples() const
	{
		return m_numberOfClassExamples;
	}

	void MultipleGammaMemeticTrainingSetAdaptation::resetToInitial(unsigned int numberOfClassExamples)
	{
		m_isLocalMode = m_isLocalModeInitialValue;
		m_numberOfClassExamples = (numberOfClassExamples);
		m_currentIteration = 0;
		m_previousModeFitness = 0;
		m_previousIterationFitness = 0;
	}

	void MultipleGammaMemeticTrainingSetAdaptation::setFrozenSetSize(unsigned int size)
	{
		m_frozenOnesSize = size;
	}

	MultipleGammaMemeticTrainingSetAdaptation::MultipleGammaMemeticTrainingSetAdaptation(bool isLocalMode,
		unsigned int numberOfClassExamples,
		platform::Percent percentOfSupportVectorsThreshold,
		unsigned int iterationsBeforeChangeThreshold,
		const std::vector<unsigned int>& classCount,
		double thresholdForMaxNumberOfClassExamples)
		: m_isLocalMode(isLocalMode)
		, m_isLocalModeInitialValue(isLocalMode)
		, m_numberOfClassExamples(numberOfClassExamples)
		, m_currentIteration(0)
		, m_previousModeFitness(0)
		, m_previousIterationFitness(0)
		, m_frozenOnesSize(0)
		, m_percentOfSupportVectorsThreshold(percentOfSupportVectorsThreshold)
		, m_iterationsBeforeChangeThreshold(iterationsBeforeChangeThreshold)
		, m_maxNumberOfClassExamples(static_cast<unsigned int>(*std::min_element(classCount.begin(), classCount.end()) * thresholdForMaxNumberOfClassExamples))
	{
		validate();
	}

	void MultipleGammaMemeticTrainingSetAdaptation::validate() const
	{
		if (m_numberOfClassExamples == 0 || m_numberOfClassExamples > m_maxNumberOfClassExamples)
		{
			constexpr auto minimumNumberOfExamplesPerClass = 1u;
			throw ValueNotInRange("numberOfClassExamples",
				m_numberOfClassExamples,
				minimumNumberOfExamplesPerClass,
				m_maxNumberOfClassExamples);
		}
		if (m_iterationsBeforeChangeThreshold == 0)
		{
			constexpr auto minimumNumberOfIteration = 1u;
			throw ValueNotInRange("iterationsBeforeChangeThreshold",
				m_iterationsBeforeChangeThreshold,
				minimumNumberOfIteration,
				std::numeric_limits<unsigned int>::max());
		}
	}

	void MultipleGammaMemeticTrainingSetAdaptation::growSizeOfTraningSet(double bestOneFitness, platform::Percent percentOfSupportVectors)
	{
		auto growingFactor = 1 + std::abs(percentOfSupportVectors.m_percentValue - m_percentOfSupportVectorsThreshold.m_percentValue) /
			(1 - m_percentOfSupportVectorsThreshold.m_percentValue);
		if (growingFactor * m_numberOfClassExamples < m_maxNumberOfClassExamples)
		{
			m_numberOfClassExamples = static_cast<unsigned int>(growingFactor * m_numberOfClassExamples);
		}
		m_previousModeFitness = bestOneFitness;
		m_isLocalMode = true;
		m_currentIteration = 0;
	}

	bool MultipleGammaMemeticTrainingSetAdaptation::adaptationCondition(double deltaIteration,
		double deltaMode,
		platform::Percent percentOfSupportVectors,
		double improvementRate) const
	{
		return m_currentIteration < m_iterationsBeforeChangeThreshold ||
			deltaIteration >= improvementRate * deltaMode ||
			percentOfSupportVectors < m_percentOfSupportVectorsThreshold ||
			deltaMode == 0;
	}

	void MultipleGammaMemeticTrainingSetAdaptation::adapt(geneticComponents::Population<SvmCustomKernelChromosome>& population)
	{
		auto& bestOne = population.getBestOne();
		if (bestOne.getClassifier() && bestOne.getClassifier()->isTrained())
		{
			auto bestOneFitness = bestOne.getFitness();
			auto deltaIteration = bestOneFitness - m_previousIterationFitness;
			auto deltaMode = bestOneFitness - m_previousModeFitness;

			auto percentOfSupportVectors = platform::Percent(static_cast<double>(bestOne.getNumberOfSupportVectors()) / static_cast<double>(bestOne.getDataset().size() + m_frozenOnesSize));
			constexpr auto improvementRate = 0.5;

			if (adaptationCondition(deltaIteration, deltaMode, percentOfSupportVectors, improvementRate))
			{
				++m_currentIteration;
			}
			else if (m_isLocalMode)
			{
				m_isLocalMode = false;
			}
			else
			{
				growSizeOfTraningSet(bestOneFitness, percentOfSupportVectors);
			}
			m_previousIterationFitness = bestOneFitness;
			return;
		}
		//throw UntrainedSvmClassifierException();
	}


	MultipleGammaCompensationInformation::MultipleGammaCompensationInformation(std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator,
		unsigned int numberOfClasses)
		: m_rngEngine(std::move(randomNumberGenerator))
		, m_numberOfClasses(numberOfClasses)
	{
		if (m_rngEngine == nullptr)
		{
			throw RandomNumberGeneratorNullPointer();
		}
	}

	std::vector<unsigned int> MultipleGammaCompensationInformation::generate(const std::vector<Parents<SvmCustomKernelChromosome>>& parents,
		unsigned int numberOfClassExamples) const
	{
		std::vector<unsigned int> compensationInfo(parents.size());

		std::transform(parents.begin(), parents.end(), compensationInfo.begin(), [&, this](const auto& parentsPair)
			{
				auto parentsMax = static_cast<int>(std::max(parentsPair.first.getDataset().size(), parentsPair.second.getDataset().size()));
				auto min = static_cast<int>(std::min(parentsMax, static_cast<int>(m_numberOfClasses * numberOfClassExamples)));
				auto max = static_cast<int>(std::max(parentsMax, static_cast<int>(m_numberOfClasses * numberOfClassExamples)));

				std::uniform_int_distribution<int> sizeOfChild(min, max);
				auto newSize = m_rngEngine->getRandom(sizeOfChild);

				if (newSize - parentsMax < 0)
					return 0u;

				return static_cast<unsigned int>(newSize - parentsMax);
			});
		return compensationInfo;
	}






	MultipleGammaMutation::MultipleGammaMutation(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
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
		//m_gamma = 0;
		m_forbiddenIds = {};
	}

	void MultipleGammaMutation::mutatePopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population)
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

	std::unordered_set<uint64_t> MultipleGammaMutation::setDifference(const std::vector<Gene>& set,
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

	void MultipleGammaMutation::calculateNumberOfPossibleExchanges(SvmCustomKernelChromosome& chromosome,
		std::vector<uint64_t>& possibleNumberOfExchangesPerClass,
		const std::vector<uint64_t>& forbiddenExchangesNumber) const
	{
		for (auto i = 0u; i < m_numberOfClasses; ++i)
		{
			possibleNumberOfExchangesPerClass[i] = m_labelsCount[i] - forbiddenExchangesNumber[i] - (chromosome.size() / m_numberOfClasses);
		}
	}

	std::vector<Gene> MultipleGammaMutation::findReplacement(const std::unordered_set<uint64_t>& deleted,
		std::unordered_set<uint64_t>& mutated,
		const std::vector<std::size_t>& positionsToReplace,
		SvmCustomKernelChromosome& chromosome) const
	{
		std::uniform_int_distribution<int> datasetPosition(0, static_cast<int>(m_trainingSet.size() - 1));
		const auto targets = m_trainingSet.getLabels();
		auto dataset = chromosome.getDataset();

		//
		std::vector<uint64_t> forbiddenExchangesNumber(m_numberOfClasses);
		for (auto& id : m_forbiddenIds)
		{
			forbiddenExchangesNumber[static_cast<int>(targets[id])]++;
		}

		std::vector<uint64_t> possibleNumberOfExchangesPerClass(m_numberOfClasses);
		calculateNumberOfPossibleExchanges(chromosome, possibleNumberOfExchangesPerClass, forbiddenExchangesNumber);

		//gammas
		auto gammaValueRandom = std::uniform_int_distribution<int>(0, static_cast<int>(m_gamma.size() - 1));

		for (auto i = 0u; i < m_numberOfExchanges; i++)
		{
			auto j = 0;
			while (true && j < 10000) //TODO fix this hack
			{
				auto newId = m_rngEngine->getRandom(datasetPosition);
				if (static_cast<int>(dataset[positionsToReplace[i]].classValue) == targets[newId] && // @wdudzik class value match
					deleted.find(newId) == deleted.end() &&
					mutated.emplace(newId).second && // @wdudzik is newId unique in chromosome dataset
					(m_forbiddenIds.empty() || m_forbiddenIds.find(newId) == m_forbiddenIds.end())) // not in forbidden
				{
					dataset[positionsToReplace[i]].id = newId;
					dataset[positionsToReplace[i]].gamma = m_gamma[m_rngEngine->getRandom(gammaValueRandom)];
					possibleNumberOfExchangesPerClass[static_cast<int>(dataset[positionsToReplace[i]].classValue)]--;
					break;
				}
				if (possibleNumberOfExchangesPerClass[static_cast<int>(dataset[positionsToReplace[i]].classValue)] == 0)
				{
					break;
				}
				j++;
			}
		}

		std::uniform_int_distribution<int> chromosomePosition(0, static_cast<int>(dataset.size() - 1));
		for (auto i = 0u; i < m_numberOfExchanges; i++)
		{
			{
				auto randomID = m_rngEngine->getRandom(chromosomePosition);
				dataset[randomID].gamma = m_gamma[m_rngEngine->getRandom(gammaValueRandom)];
				
			}
		}
		
		return dataset;
	}

	inline void MultipleGammaMutation::getPositionsOfMutation(SvmCustomKernelChromosome& chromosome,
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

	void MultipleGammaMutation::mutateChromosome(SvmCustomKernelChromosome& chromosome)
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






} // namespace svmComponents
