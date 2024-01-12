
#pragma once

#include <libDataset/Dataset.h>
#include "SvmTrainingSetChromosome.h"


namespace svmComponents
{
class SvmCustomKernelChromosome;
struct Gene;

class ISupportVectorSelection
{
public:
    virtual ~ISupportVectorSelection() = default;

    virtual void addSupportVectors(const SvmTrainingSetChromosome& chromosome,
                                   const dataset::Dataset<std::vector<float>, float>& trainingSet) = 0;

    virtual const std::vector<DatasetVector>& getSupportVectorPool() const = 0;
    virtual const std::unordered_set<uint64_t>& getSupportVectorIds() const = 0;
};


class ISupportVectorSelectionGamma
{
public:
	virtual ~ISupportVectorSelectionGamma() = default;

	virtual void addSupportVectors(const SvmCustomKernelChromosome& chromosome,
	                               const dataset::Dataset<std::vector<float>, float>& trainingSet) = 0;

	virtual const std::vector<Gene>& getSupportVectorPool() const = 0;
	virtual const std::unordered_set<uint64_t>& getSupportVectorIds() const = 0;
};
} // namespace svmComponents
