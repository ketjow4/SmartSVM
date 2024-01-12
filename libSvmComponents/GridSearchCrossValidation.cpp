
#include "FoldCreator.h"
#include "SvmAccuracyMetric.h"
#include "GridSearchCrossValidation.h"

namespace svmComponents
{
using namespace geneticComponents;

GridSearchCrossValidation::GridSearchCrossValidation(GridSearchConfiguration& algorithmConfig)
    : m_algorithmConfig(algorithmConfig)
    , m_trainingMethod(m_algorithmConfig.m_svmConfig, m_algorithmConfig.m_svmConfig.m_estimationType == svmMetricType::Auc)
{
}

void GridSearchCrossValidation::validate(Population<SvmKernelChromosome>& population,
                                         std::vector<double>& results,
                                         const dataset::Dataset<std::vector<float>, float>& validationSet)
{
    const size_t iterationCount = std::distance(population.begin(), population.end());
    auto first = population.begin();

#pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < static_cast<int>(iterationCount); i++)
    {
        auto& individual = *(first + i);
        auto fitnessMatrixPair = m_algorithmConfig.m_svmConfig.m_estimationMethod->calculateMetric(individual, validationSet, false);
        individual.updateFitness(fitnessMatrixPair.m_fitness);
        individual.updateConfusionMatrix(fitnessMatrixPair.m_confusionMatrix);
        results[i] += individual.getFitness();
    }
}

void GridSearchCrossValidation::trainClassifiers(Population<SvmKernelChromosome>& population,
                                       const dataset::Dataset<std::vector<float>, float>& trainingDataset,
                                       std::vector<double>& results)
{
    FoldCreator split{m_algorithmConfig.m_numberOfFolds, trainingDataset};

	if(m_algorithmConfig.m_numberOfFolds == 1)
	{
		const auto[_, test] = split.getFold(0);

		m_trainingMethod.trainPopulation(population, test);
		validate(population, results, test);	//when there is 1 fold training is empty (name are wrong inside split)
		return; 
	}

    for (auto i = 0u; i < m_algorithmConfig.m_numberOfFolds; ++i)
    {
        const auto [training, test] = split.getFold(i);

        m_trainingMethod.trainPopulation(population, training);
        validate(population, results, test);	
    }
}

void GridSearchCrossValidation::averageResults(std::vector<double>& results, Population<SvmKernelChromosome>& population)
{
    std::transform(results.begin(), results.end(), results.begin(),
                   [this](auto fitnessValue)
               {
                   return fitnessValue / static_cast<double>(m_algorithmConfig.m_numberOfFolds);
               });
    auto i = 0;
    for(auto& individual : population)
    {
        individual.updateFitness(results[i]);
        i++;
    }
}

SvmKernelChromosome GridSearchCrossValidation::findBestParameters(const Population<SvmKernelChromosome>& population,
                                                                  const std::vector<double>& results) 
{
    const auto bestIterator = std::max_element(results.begin(), results.end());
    const auto bestIndex = std::distance(results.begin(), bestIterator);
    return population[static_cast<int>(bestIndex)];
}

Population<SvmKernelChromosome> GridSearchCrossValidation::run(Population<SvmKernelChromosome>& population,
                                                               const dataset::Dataset<std::vector<float>, float>& trainingDataset) 
{
    std::vector<double> results(population.size());

    trainClassifiers(population, trainingDataset, results);
    averageResults(results, population);
    auto bestOne = findBestParameters(population, results);
    Population<SvmKernelChromosome> bestParameters{std::vector<SvmKernelChromosome>{std::move(bestOne)}};

    m_trainingMethod.trainPopulation(bestParameters, trainingDataset);

    return population;
}
} // namespace svmComponents
