
#pragma once

#include "LibGeneticComponents/Population.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libPlatform/Percent.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class MemeticTrainingSetAdaptation
{
public:
    MemeticTrainingSetAdaptation(bool isLocalMode,
                                 unsigned int numberOfClassExamples,
                                 platform::Percent percentOfSupportVectorsThreshold,
                                 unsigned int iterationsBeforeChangeThreshold,
                                 const std::vector<unsigned int>& classCount,
                                 double thresholdForMaxNumberOfClassExamples);

    MemeticTrainingSetAdaptation(bool isLocalMode,
        unsigned int numberOfClassExamples,
        platform::Percent percentOfSupportVectorsThreshold,
        unsigned int iterationsBeforeChangeThreshold,
        const std::vector<unsigned int>& classCount,
        double thresholdForMaxNumberOfClassExamples,
        double maxK);
	
    void adapt(geneticComponents::Population<SvmTrainingSetChromosome>& population);

    bool getIsModeLocal() const;
    unsigned int getNumberOfClassExamples() const;

    void setK(unsigned int /*numberOfClassExamples*/)
    {
        m_numberOfClassExamples = m_numberOfClassExamples * 2;
        m_maxNumberOfClassExamples = static_cast<unsigned int>(*std::min_element(m_classCount.begin(), m_classCount.end()));
    }

    std::shared_ptr<phd::svm::ISvm> getClassifierWithBestDistance() const
    {
        return m_DistancesBest;
    }

    void setFrozenSetSize(unsigned int size) //only for usage with DA-SVM kernels see SvmHelper for usage
    {
        m_frozenOnesSize = size;
    }
	
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
    unsigned int m_maxNumberOfClassExamples;
    const std::vector<unsigned> m_classCount;

    std::shared_ptr<phd::svm::ISvm> m_DistancesBest;
};

inline bool MemeticTrainingSetAdaptation::getIsModeLocal() const
{
    return m_isLocalMode;
}

inline unsigned int MemeticTrainingSetAdaptation::getNumberOfClassExamples() const
{
    return m_numberOfClassExamples;
}
} // namespace svmComponents
