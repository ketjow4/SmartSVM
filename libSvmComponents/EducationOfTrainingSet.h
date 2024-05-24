
#pragma once
#include <memory>
#include "LibGeneticComponents/ICrossoverSelection.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/SupportVectorPool.h"
#include "libSvmComponents/SvmUtils.h"
#include "libPlatform/Percent.h"

namespace svmComponents
{
// @wdudzik implementation based on: 2014 Nalepa and Kawulok - A memetic algorithm to select training data for SVMs
class EducationOfTrainingSet
{
public:
    virtual ~EducationOfTrainingSet() = default;

    EducationOfTrainingSet(platform::Percent educationProbability,
                           unsigned int numberOfClasses,
                           std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator,
                           std::unique_ptr<ISupportVectorSelection> supportVectorSelection);

    void educatePopulation(geneticComponents::Population<SvmTrainingSetChromosome>& population,
                           const std::vector<DatasetVector>& supportVectorPool,
                           const std::vector<geneticComponents::Parents<SvmTrainingSetChromosome>>& parents,
                           const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

    void educate(SvmTrainingSetChromosome& individual,
                 const geneticComponents::Parents<SvmTrainingSetChromosome>& parents,
                 const std::vector<DatasetVector>& supportVectorPool,
                 const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

private:
    static bool replacementCondition(const DatasetVector& supportVectorPoolElement,
                                                             std::unordered_set<std::uint64_t>& traningIDs,
                                                             const DatasetVector& sample);

    static std::vector<DatasetVector> setDifference(const std::unordered_set<uint64_t>& svPool,
                                                    const std::vector<DatasetVector>& traningDataset);

    std::unordered_set<uint64_t> findSupportVectors(const geneticComponents::Parents<SvmTrainingSetChromosome>& parents,
                                                    const dataset::Dataset<std::vector<float>, float>& trainingSet) const;

    platform::Percent m_educationProbability;
    const unsigned int m_numberOfClasses;
    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    std::unique_ptr<ISupportVectorSelection> m_supportVectorSelection;
};
} // namespace svmComponents
