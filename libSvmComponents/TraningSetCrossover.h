
#pragma once

#include <memory>
#include <unordered_set>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"

namespace svmComponents
{
// @wdudzik Common part of crossover operators from GaSvm and Memetic algorithms
class TrainingSetCrossover : public geneticComponents::BaseCrossoverOperator<SvmTrainingSetChromosome>
{
public:
    using chromosomeType = SvmTrainingSetChromosome;

protected:
    explicit TrainingSetCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                  unsigned int numberOfClasses);

    virtual void tryToInsert(const std::vector<DatasetVector>::const_iterator& parentAIt,
                     const std::vector<DatasetVector>::const_iterator& parentBIt,
                     std::unordered_set<uint64_t>& childSet,
                     std::vector<unsigned>& classes,
                     unsigned int classExamples,
                     std::vector<DatasetVector>& child) const;
    std::vector<DatasetVector> crossoverInternals(const chromosomeType& parentA,
                                                  const chromosomeType& parentB,
                                                  unsigned int datasetSize,
                                                  unsigned int classExamples);

    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClasses;
};
} // namespace svmComponents
