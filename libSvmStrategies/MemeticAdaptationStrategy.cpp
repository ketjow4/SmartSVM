#include "libGeneticComponents/Population.h"
#include "MemeticAdaptationStrategy.h"

namespace svmStrategies
{
MemeticAdaptationStrategy::MemeticAdaptationStrategy(svmComponents::MemeticTrainingSetAdaptation& adaptation)
	: m_adaptation(adaptation)
{
}

std::string MemeticAdaptationStrategy::getDescription() const
{
	return "Performs adaptation of memetic algorithm for training set selection";
}

std::tuple<bool, unsigned int> MemeticAdaptationStrategy::launch(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population)
{
	m_adaptation.adapt(population);

	return std::make_tuple(m_adaptation.getIsModeLocal(), m_adaptation.getNumberOfClassExamples());
}

void MemeticAdaptationStrategy::setFrozenSetSize(unsigned size)
{
	m_adaptation.setFrozenSetSize(size);
}
} // namespace svmStrategies
