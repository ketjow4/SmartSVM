

#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "GeneticAlgorithmsConfigs.h"

namespace svmComponents
{
enum class SvmTrainingSetCrossover
{
    Unknown,
    GaSvm,
    Memetic,
    GaSvmRegression,
	EnhanceTrainingSet,
};

class SvmTrainingSetCrossoverFactory
{
public:
    static CrossoverOperator<SvmTrainingSetChromosome> create(const platform::Subtree& config, std::shared_ptr<ITrainingSet> enhanceTrainingSet);
private:
    const static std::unordered_map<std::string, SvmTrainingSetCrossover> m_translations;
};
} // namespace svmComponents
