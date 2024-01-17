#pragma once

#include "libSvmComponents/MemeticTraningSetAdaptation.h"

namespace svmStrategies
{
class MemeticAdaptationStrategy
{
public:
	explicit MemeticAdaptationStrategy(svmComponents::MemeticTrainingSetAdaptation& adaptation);

	std::string getDescription() const;
	std::tuple<bool, unsigned int> launch(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population);

	void setFrozenSetSize(unsigned int size); //only for usage with DA-SVM kernels see SvmHelper for usage

private:
	svmComponents::MemeticTrainingSetAdaptation& m_adaptation;
};
} // namespace svmStrategies
