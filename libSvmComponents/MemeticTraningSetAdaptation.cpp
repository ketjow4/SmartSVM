
#include "MemeticTraningSetAdaptation.h"
#include "SvmComponentsExceptions.h"
#include "SvmUtils.h"

namespace svmComponents
{
MemeticTrainingSetAdaptation::MemeticTrainingSetAdaptation(bool isLocalMode,
                                                           unsigned int numberOfClassExamples,
                                                           platform::Percent percentOfSupportVectorsThreshold,
                                                           unsigned int iterationsBeforeChangeThreshold,
                                                           const std::vector<unsigned int>& classCount,
                                                           double thresholdForMaxNumberOfClassExamples)
    : m_isLocalMode(isLocalMode)
    , m_numberOfClassExamples(numberOfClassExamples)
    , m_currentIteration(0)
    , m_previousModeFitness(0)
    , m_previousIterationFitness(0)
    , m_frozenOnesSize(0)
    , m_percentOfSupportVectorsThreshold(percentOfSupportVectorsThreshold)
    , m_iterationsBeforeChangeThreshold(iterationsBeforeChangeThreshold)
	, m_maxNumberOfClassExamples(static_cast<unsigned int>(*std::min_element(classCount.begin(), classCount.end()) * thresholdForMaxNumberOfClassExamples))
    , m_classCount(classCount)
{
    validate();
}

MemeticTrainingSetAdaptation::MemeticTrainingSetAdaptation(bool isLocalMode, unsigned numberOfClassExamples, platform::Percent percentOfSupportVectorsThreshold,
	unsigned iterationsBeforeChangeThreshold, const std::vector<unsigned>& classCount, double thresholdForMaxNumberOfClassExamples, double maxK)
    : m_isLocalMode(isLocalMode)
    , m_numberOfClassExamples(numberOfClassExamples)
    , m_currentIteration(0)
    , m_previousModeFitness(0)
    , m_previousIterationFitness(0)
    , m_frozenOnesSize(0)
    , m_percentOfSupportVectorsThreshold(percentOfSupportVectorsThreshold)
    , m_iterationsBeforeChangeThreshold(iterationsBeforeChangeThreshold)
	, m_maxNumberOfClassExamples(static_cast<unsigned int>(*std::min_element(classCount.begin(), classCount.end())* thresholdForMaxNumberOfClassExamples))
	, m_classCount(classCount)
{
    if(maxK != 0)
    {
        m_maxNumberOfClassExamples = static_cast<unsigned int>(maxK);
    }
	
    validate();
}


void MemeticTrainingSetAdaptation::validate() const
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

void MemeticTrainingSetAdaptation::growSizeOfTraningSet(double bestOneFitness, platform::Percent percentOfSupportVectors)
{
    auto growingFactor = 1 + std::abs(percentOfSupportVectors.m_percentValue - m_percentOfSupportVectorsThreshold.m_percentValue) /
            (1 - m_percentOfSupportVectorsThreshold.m_percentValue);
    if (growingFactor * m_numberOfClassExamples < m_maxNumberOfClassExamples)
    {
        std::cout << "Growing number of SV: " << static_cast<unsigned int>(growingFactor * m_numberOfClassExamples) << "   Max: " << m_maxNumberOfClassExamples << "\n";
        std::cout << "Class count: " << m_classCount[0] << ", " << m_classCount[1] << "\n";
    	
        m_numberOfClassExamples = static_cast<unsigned int>(growingFactor * m_numberOfClassExamples);
    }
    else
    {
        std::cout << "Growing to MAX: " <<  m_maxNumberOfClassExamples << "\n";
        std::cout << "Class count: " << m_classCount[0] << ", " << m_classCount[1] << "\n";

        m_numberOfClassExamples = m_maxNumberOfClassExamples;
    }
	
    m_previousModeFitness = bestOneFitness;
    m_isLocalMode = true;
    m_currentIteration = 0;
}

bool MemeticTrainingSetAdaptation::adaptationCondition(double deltaIteration,
                                                       double deltaMode,
                                                       platform::Percent percentOfSupportVectors,
                                                       double improvementRate) const
{
    return m_currentIteration < m_iterationsBeforeChangeThreshold ||
            deltaIteration >= improvementRate * deltaMode ||
            percentOfSupportVectors < m_percentOfSupportVectorsThreshold ||
            deltaMode == 0;
}

void MemeticTrainingSetAdaptation::adapt(geneticComponents::Population<SvmTrainingSetChromosome>& population)
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
        	//save the best Distance from individual
            m_DistancesBest = population.getBestOne().getClassifier();
            growSizeOfTraningSet(bestOneFitness, percentOfSupportVectors);
        }
        m_previousIterationFitness = bestOneFitness;
        return;
    }
    throw UntrainedSvmClassifierException();
}
} // namespace svmComponents
