
#include <unordered_set>
#include "libGeneticComponents/Population.h"
#include "GaSvmGeneration.h"
#include "SvmComponentsExceptions.h"

namespace svmComponents
{
GaSvmGeneration::GaSvmGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                 std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                 unsigned int numberOfClassExamples,
                                 const std::vector<unsigned int>& labelsCount)
    : m_trainingSet(trainingSet)
    , m_rngEngine(std::move(rngEngine))
    , m_numberOfClassExamples(numberOfClassExamples)
    , m_numberOfClasses(static_cast<unsigned int>(labelsCount.size()))
{
    if (std::any_of(labelsCount.begin(), labelsCount.end(), [this](const auto& labelCount) { return m_numberOfClassExamples > labelCount;  }))
    {
        throw ValueOfClassExamplesIsTooHighForDataset(m_numberOfClassExamples);
    }
}

geneticComponents::Population<SvmTrainingSetChromosome> GaSvmGeneration::createPopulation(uint32_t populationSize)
{
    if (populationSize == 0)
    {
        throw geneticComponents::PopulationIsEmptyException();
    }

    auto targets = m_trainingSet.getLabels();
    auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));
    std::vector<SvmTrainingSetChromosome> population(populationSize);
    
    std::generate(population.begin(), population.end(), [&]
    {
        std::unordered_set<std::uint64_t> trainingSet;
        std::vector<DatasetVector> chromosomeDataset;
        chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
        std::vector<unsigned int> classCount(m_numberOfClasses, 0);
        while (std::any_of(classCount.begin(), classCount.end(), [this](const auto& classIndicies) { return classIndicies != m_numberOfClassExamples;  }))
        {
            auto randomValue = m_rngEngine->getRandom(trainingSetID);
            if (classCount[static_cast<int>(targets[randomValue])] < m_numberOfClassExamples &&    // less that desired number of class examples
                trainingSet.emplace(static_cast<int>(randomValue)).second)       // is unique
            {
                chromosomeDataset.emplace_back(DatasetVector(randomValue, static_cast<std::uint8_t>(targets[randomValue])));
                classCount[static_cast<int>(targets[randomValue])]++;
            }
        }
        return SvmTrainingSetChromosome(std::move(chromosomeDataset));
    });
    return geneticComponents::Population<SvmTrainingSetChromosome>(std::move(population));
}


GaSvmGenerationWithForbbidenSet::GaSvmGenerationWithForbbidenSet(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
	unsigned numberOfClassExamples, const std::vector<unsigned>& labelsCount)
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
		m_forbiddenIds = {};
		m_imbalancedOrOneClass = false;
}

geneticComponents::Population<SvmTrainingSetChromosome> GaSvmGenerationWithForbbidenSet::createPopulation(uint32_t populationSize)
{
	//populationSize = 20;
	if (populationSize == 0)
	{
		throw geneticComponents::PopulationIsEmptyException();
	}

	auto samples = m_trainingSet.getSamples();
	auto targets = m_trainingSet.getLabels();
	auto trainingSetID = std::uniform_int_distribution<int>(0, static_cast<int>(m_trainingSet.size() - 1));
	std::vector<SvmTrainingSetChromosome> population(populationSize);

	std::generate(population.begin(), population.end(), [&]
		{
			//auto it = trainingSetOnce.begin();
			std::unordered_set<std::uint64_t> trainingSet;
			std::vector<DatasetVector> chromosomeDataset;
			chromosomeDataset.reserve(m_numberOfClassExamples * m_numberOfClasses);
			std::vector<unsigned int> classCount(m_numberOfClasses, 0);
			while (std::any_of(classCount.begin(), classCount.end(), [&](const auto& classIndicies)
				{
					return m_imbalancedOrOneClass ? classCount[0] != m_numberOfClassExamples : classIndicies != m_numberOfClassExamples;
				}))
			{

				auto randomValue = m_rngEngine->getRandom(trainingSetID); //*it;
				//++it;
				if ((!m_imbalancedOrOneClass
					&& classCount[static_cast<int>(targets[randomValue])] < m_numberOfClassExamples // less that desired number of class examples
					&& trainingSet.emplace(static_cast<int>(randomValue)).second // is unique
					&& (m_forbiddenIds.empty() || m_forbiddenIds.find(randomValue) == m_forbiddenIds.end()) // not in forbidden
					)
					||
					(m_imbalancedOrOneClass
						&& classCount[0] < m_numberOfClassExamples
						&& trainingSet.emplace(static_cast<int>(randomValue)).second // is unique)
						&& (m_forbiddenIds.empty() || m_forbiddenIds.find(randomValue) == m_forbiddenIds.end()))
					)
				{
					chromosomeDataset.emplace_back(DatasetVector(randomValue, static_cast<std::uint8_t>(targets[randomValue])));
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

				return SvmTrainingSetChromosome(std::move(chromosomeDataset));
		});

	return geneticComponents::Population<SvmTrainingSetChromosome>(std::move(population));
}

} // namespace svmComponents