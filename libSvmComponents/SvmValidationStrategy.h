#pragma once

#include "libGeneticComponents/Population.h"
#include "libSvmComponents/ISvmMetricsCalculator.h"
#include "libSvmComponents/BaseSvmChromosome.h"
#include "libSvmComponents/SvmFeatureSetChromosome.h"
#include "libSvmComponents/SvmFeatureSetMemeticChromosome.h"
#include "libSvmComponents/SvmSimultaneousChromosome.h"
#include "SvmCustomKernelFeaturesSelectionChromosome.h"

namespace svmStrategies
{
template <class chromosome>
class IValidationStrategy
{
public:
	virtual ~IValidationStrategy() = default;

	virtual geneticComponents::Population<chromosome>& launch(geneticComponents::Population<chromosome>& population,
		const dataset::Dataset<std::vector<float>, float>& validationSet) = 0;

    virtual bool isUsingFullSet() const = 0;

    virtual void generateNewSubset(const dataset::Dataset<std::vector<float>, float>& /*validationSet*/) { ; }
};

template <class chromosome>
class SvmValidationStrategy : public IValidationStrategy<chromosome>
{
    static_assert(std::is_base_of<svmComponents::BaseSvmChromosome, chromosome>::value, "Cannot do validation for class not derived from BaseSvmChromosome");
public:
    explicit SvmValidationStrategy(const svmComponents::ISvmMetricsCalculator& metric, bool isTestSet);

    std::string getDescription() const;
    geneticComponents::Population<chromosome>& launch(geneticComponents::Population<chromosome>& population,
                                                      const dataset::Dataset<std::vector<float>, float>& validationSet) override;

    geneticComponents::Population<chromosome>& launchSingleThread(geneticComponents::Population<chromosome>& population,
																  const dataset::Dataset<std::vector<float>, float>& validationSet);

    bool isUsingFullSet() const override { return true; }
	
private:
    using Clock = std::chrono::high_resolution_clock;

    auto updateFitness(chromosome& individual,
                       const dataset::Dataset<std::vector<float>, float>& validationSet);
	
    const svmComponents::ISvmMetricsCalculator& m_metric;
    bool m_isTestSet;
};

#pragma warning( push )
#pragma warning( disable : 4505) // @wdudzik Warning C4505 unreferenced local function, 
// there is a problem with specialization which is generated and not used (as it is normal for specialization)

template <class chromosome>
SvmValidationStrategy<chromosome>::SvmValidationStrategy(const svmComponents::ISvmMetricsCalculator& metric, bool isTestSet)
    : m_metric(metric)
	, m_isTestSet(isTestSet)
{
}

template <class chromosome>
std::string SvmValidationStrategy<chromosome>::getDescription() const
{
    return "Estimate population fitness based on given estimation method";
}

//#pragma optimize("", off)

template <class chromosome>
geneticComponents::Population<chromosome>& SvmValidationStrategy<chromosome>::launch(geneticComponents::Population<chromosome>& population,
                                                                                     const dataset::Dataset<std::vector<float>, float>& validationSet)
{
    const size_t iterationCount = std::distance(population.begin(), population.end());
    auto first = population.begin();

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(iterationCount); i++)
    {
        auto& individual = *(first + i);
        auto begin = updateFitness(individual, validationSet);
        individual.updateTime(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin));
    }

    return population;
}

template <class chromosome>
geneticComponents::Population<chromosome>& SvmValidationStrategy<chromosome>::launchSingleThread(geneticComponents::Population<chromosome>& population,
	const dataset::Dataset<std::vector<float>, float>& validationSet)
{
    const size_t iterationCount = std::distance(population.begin(), population.end());
    auto first = population.begin();

    for (int i = 0; i < static_cast<int>(iterationCount); i++)
    {
        auto& individual = *(first + i);
        auto begin = updateFitness(individual, validationSet);
        individual.updateTime(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin));
    }

    return population;
}

template <class chromosome>
auto SvmValidationStrategy<chromosome>::updateFitness(chromosome& individual, const dataset::Dataset<std::vector<float>, float>& validationSet)
{
    auto begin = Clock::now();
    const auto metric = m_metric.calculateMetric(individual, validationSet, m_isTestSet);
    individual.updateFitness(metric.m_fitness);
    individual.updateConfusionMatrix(metric.m_confusionMatrix);
    individual.updateMetric(metric);
    return begin;
}

//#pragma optimize("", on)

//template <>
//inline auto SvmValidationStrategy<svmComponents::SvmFeatureSetChromosome>::updateFitness(
//    svmComponents::SvmFeatureSetChromosome& individual,
//    const dataset::Dataset<std::vector<float>, float>& validationSet)
//{
//    const auto convertedSet = individual.convertChromosome(validationSet);
//    const auto begin = Clock::now();
//    const auto metric = m_metric.calculateMetric(individual, convertedSet, m_isTestSet);
//    individual.updateFitness(metric.m_fitness);
//    individual.updateConfusionMatrix(metric.m_confusionMatrix);
//    individual.updateMetric(metric);
//    return begin;
//}
//
//template <>
//inline auto SvmValidationStrategy<svmComponents::SvmFeatureSetMemeticChromosome>::updateFitness(
//    svmComponents::SvmFeatureSetMemeticChromosome& individual,
//    const dataset::Dataset<std::vector<float>, float>& validationSet)
//{
//    const auto convertedSet = individual.convertChromosome(validationSet);
//    const auto begin = Clock::now();
//    const auto metric = m_metric.calculateMetric(individual, convertedSet, m_isTestSet);
//    individual.updateFitness(metric.m_fitness);
//    individual.updateConfusionMatrix(metric.m_confusionMatrix);
//    individual.updateMetric(metric);
//    return begin;
//}
//
//template <>
//inline auto SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome>::updateFitness(
//    svmComponents::SvmSimultaneousChromosome& individual,
//    const dataset::Dataset<std::vector<float>, float>& validationSet)
//{
//   /* const auto convertedSet = individual.convertFeatures(validationSet);
//    const auto begin = Clock::now();
//    const auto metric = m_metric.calculateMetric(individual, convertedSet, m_isTestSet);
//    individual.updateFitness(metric.m_fitness);
//    individual.updateConfusionMatrix(metric.m_confusionMatrix);
//    individual.updateMetric(metric);
//    return begin;*/
//
//
//    //const auto convertedSet = individual.convertFeatures(validationSet);
//    const auto begin = Clock::now();
//    const auto metric = m_metric.calculateMetric(individual, validationSet, m_isTestSet);
//    individual.updateFitness(metric.m_fitness);
//    individual.updateConfusionMatrix(metric.m_confusionMatrix);
//    individual.updateMetric(metric);
//    return begin;
//}
//
//
//template <>
//inline auto SvmValidationStrategy<svmComponents::SvmCustomKernelFeaturesSelectionChromosome>::updateFitness(
//    svmComponents::SvmCustomKernelFeaturesSelectionChromosome& individual,
//    const dataset::Dataset<std::vector<float>, float>& validationSet)
//{
//    const auto convertedSet = individual.convertFeatures(validationSet);
//    const auto begin = Clock::now();
//    const auto metric = m_metric.calculateMetric(individual, convertedSet, m_isTestSet);
//    individual.updateFitness(metric.m_fitness);
//    individual.updateConfusionMatrix(metric.m_confusionMatrix);
//    return begin;
//}




#pragma warning( pop )
} // namespace svmStrategies
