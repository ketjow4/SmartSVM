#pragma once

#include <limits>
#include "LibGeneticComponents/Population.h"
#include "libPlatform/Percent.h"
#include "SvmFeatureSetMemeticChromosome.h"
#include "SvmComponentsExceptions.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticFeaturesAdaptation
{
public:
    MemeticFeaturesAdaptation(bool isLocalMode,
                              unsigned int numberOfFeatures,
                              platform::Percent percentOfSupportVectorsThreshold,
                              unsigned int iterationsBeforeChangeThreshold,
                              unsigned int featuresCount,
                              double thresholdForMaxNumberOfClassExamples);
    void adapt(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population);

    bool getIsModeLocal() const;
    unsigned int getNumberOfClassExamples() const;

private:
    void validate() const;
    void growSizeOfTraningSet(double bestOneFitness, platform::Percent percentOfSupportVectors);
    bool adaptationCondition(double deltaIteration,
                             double deltaMode,
                             platform::Percent percentOfSupportVectors,
                             double improvementRate) const;

    bool m_isLocalMode;
    unsigned int m_numberOfFeatures;
    unsigned int m_currentIteration;
    double m_previousModeFitness;
    double m_previousIterationFitness;

    platform::Percent m_percentOfSupportVectorsThreshold;
    const unsigned m_iterationsBeforeChangeThreshold;
    const unsigned int m_maxNumberOfFeatures;
};

inline MemeticFeaturesAdaptation::MemeticFeaturesAdaptation(bool isLocalMode, unsigned numberOfFeatures,
                                                            platform::Percent percentOfSupportVectorsThreshold, 
                                                            unsigned iterationsBeforeChangeThreshold,
                                                            unsigned int featuresCount,
                                                            double thresholdForMaxNumberOfClassExamples)
    : m_isLocalMode(isLocalMode)
    , m_numberOfFeatures(numberOfFeatures)
    , m_currentIteration(0)
    , m_previousModeFitness(0)
    , m_previousIterationFitness(0)
    , m_percentOfSupportVectorsThreshold(percentOfSupportVectorsThreshold)
    , m_iterationsBeforeChangeThreshold(iterationsBeforeChangeThreshold)
    , m_maxNumberOfFeatures(static_cast<unsigned int>(featuresCount * thresholdForMaxNumberOfClassExamples))
{
    validate();
}

inline void MemeticFeaturesAdaptation::adapt(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population)
{
    auto& bestOne = population.getBestOne();
    if (bestOne.getClassifier() && bestOne.getClassifier()->isTrained())
    {
        auto bestOneFitness = bestOne.getFitness();
        auto deltaIteration = bestOneFitness - m_previousIterationFitness;
        auto deltaMode = bestOneFitness - m_previousModeFitness;

        auto percentOfFeaturesSelected = platform::Percent(
            static_cast<double>(bestOne.size()) / static_cast<double>(m_numberOfFeatures));
        constexpr auto improvementRate = 0.5;

        if (adaptationCondition(deltaIteration, deltaMode, percentOfFeaturesSelected, improvementRate))
        {
            ++m_currentIteration;
        }
        else if (m_isLocalMode)
        {
            m_isLocalMode = false;
        }
        else
        {
            growSizeOfTraningSet(bestOneFitness, percentOfFeaturesSelected);
        }
        m_previousIterationFitness = bestOneFitness;
        return;
    }
    throw UntrainedSvmClassifierException();
}

inline void MemeticFeaturesAdaptation::validate() const
{
   /* if (m_numberOfFeatures == 0 || m_numberOfFeatures > m_maxNumberOfFeatures)
    {
        constexpr auto minimumNumberOfExamplesPerClass = 1u;
        throw ValueNotInRange("numberOfClassExamples",
                              m_numberOfFeatures,
                              minimumNumberOfExamplesPerClass,
                              m_maxNumberOfFeatures);
    }
    if (m_iterationsBeforeChangeThreshold == 0)
    {
        constexpr auto minimumNumberOfIteration = 1u;
        throw ValueNotInRange("iterationsBeforeChangeThreshold",
                              m_iterationsBeforeChangeThreshold,
                              minimumNumberOfIteration,
                              std::numeric_limits<unsigned int>::max());
    }*/
}

inline void MemeticFeaturesAdaptation::growSizeOfTraningSet(double bestOneFitness, platform::Percent percentOfFeaturesSelected)
{
    auto growingFactor = 1 + std::abs(percentOfFeaturesSelected.m_percentValue - m_percentOfSupportVectorsThreshold.m_percentValue) /
            (1 - m_percentOfSupportVectorsThreshold.m_percentValue);
    if (growingFactor * m_numberOfFeatures < m_maxNumberOfFeatures)
    {
        m_numberOfFeatures = static_cast<unsigned int>(growingFactor * m_numberOfFeatures);
    }
    m_previousModeFitness = bestOneFitness;
    m_isLocalMode = true;
    m_currentIteration = 0;
}

inline bool MemeticFeaturesAdaptation::adaptationCondition(double deltaIteration, double deltaMode, platform::Percent percentOfFeaturesSelected,
                                                           double improvementRate) const
{
    return m_currentIteration < m_iterationsBeforeChangeThreshold ||
        deltaIteration >= improvementRate * deltaMode ||
        percentOfFeaturesSelected < m_percentOfSupportVectorsThreshold  || deltaMode == 0;
}

inline bool MemeticFeaturesAdaptation::getIsModeLocal() const
{
    return m_isLocalMode;
}

inline unsigned int MemeticFeaturesAdaptation::getNumberOfClassExamples() const
{
    return m_numberOfFeatures;
}
} // namespace svmComponents
