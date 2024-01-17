
#pragma once

#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "SvmLib/ISvm.h"

namespace genetic
{
template <class chromosome>
class IGeneticWorkflow
{
public:
    virtual ~IGeneticWorkflow() = default;
    virtual void initialize() = 0;
    virtual void runGeneticAlgorithm() = 0;
    virtual chromosome getBestChromosomeInGeneration() const = 0;
    virtual geneticComponents::Population<chromosome> getPopulation() const = 0;
    virtual GeneticWorkflowResultLogger& getResultLogger() = 0;


    virtual void setTimer(std::shared_ptr<Timer> /*timer*/) {};
    virtual std::shared_ptr<Timer> getTimer( /*timer*/) { return nullptr; };


    virtual void setParents(const std::vector<geneticComponents::Parents<chromosome>>& parents)
    {
        m_parents = std::make_shared<std::vector<geneticComponents::Parents<chromosome>>>(parents);
    }

	
    //for simultaneous workflow
    virtual geneticComponents::Population<chromosome> initNoEvaluate(int popSize) = 0;
    virtual geneticComponents::Population<chromosome> initNoEvaluate(int popSize, int seed) = 0;
    virtual void performGeneticOperations(geneticComponents::Population<chromosome>& population) = 0;

protected:
    std::shared_ptr<std::vector<geneticComponents::Parents<chromosome>>> m_parents;
};

template <class chromosome>
class ITrainingSetOptimizationWorkflow : public IGeneticWorkflow<chromosome>
{
public:
    virtual ~ITrainingSetOptimizationWorkflow() = default;

    virtual void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) = 0;
    virtual void setupFeaturesSet(const svmComponents::SvmFeatureSetChromosome& featureSetChromosome) = 0;

    virtual unsigned int getInitialTrainingSetSize() = 0;
    virtual unsigned int getCurrentTrainingSetSize() = 0;
    virtual void setK(unsigned int k) = 0;
    virtual phd::svm::ISvm& getClassifierWithBestDistances() = 0;
    virtual dataset::Dataset<std::vector<float>, float> getBestTrainingSet() const = 0;
};

template <class chromosome>
class IKernelOptimalizationWorkflow : public IGeneticWorkflow<chromosome>
{
public:
    virtual ~IKernelOptimalizationWorkflow() = default;
    virtual void setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet) = 0;

    virtual void setDatasets(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                             const dataset::Dataset<std::vector<float>, float>& validationSet,
                             const dataset::Dataset<std::vector<float>, float>& testSet) = 0;

};

template <class chromosome>
class IFeatureSelectionWorkflow : public IGeneticWorkflow<chromosome>
{
public:
    virtual ~IFeatureSelectionWorkflow() = default;
    virtual void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) = 0;
    virtual dataset::Dataset<std::vector<float>, float> getFilteredTraningSet() = 0;
    virtual dataset::Dataset<std::vector<float>, float> getFilteredValidationSet() = 0;
    virtual dataset::Dataset<std::vector<float>, float> getFilteredTestSet() = 0;
    virtual void setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet) = 0;
};
} // namespace genetic
