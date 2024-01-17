#pragma once

#include "libSvmComponents/MemeticFeaturesEducation.h"
#include "libSvmComponents/MemeticFeaturesAdaptation.h"
#include "libSvmComponents/MemeticFeaturesCompensationGeneration.h"
#include "libSvmComponents/MemeticFeatureCompensation.h"
#include "libSvmComponents/MemeticFeaturesPool.h"

namespace svmStrategies
{
class MemeticFeaturesEducationStrategy
{
public:
    explicit MemeticFeaturesEducationStrategy(svmComponents::MemeticFeaturesEducation& educationAlgorithm)
        : m_educationOfTrainingSet(educationAlgorithm)
    {}

    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& launch(
        geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
        std::vector<svmComponents::Feature>& supportVectorPool,
        std::vector<geneticComponents::Parents<svmComponents::SvmFeatureSetMemeticChromosome>>& parents,
        const dataset::Dataset<std::vector<float>, float>& trainingSet)
    {
        m_educationOfTrainingSet.educatePopulation(population, supportVectorPool, parents, trainingSet);

        return population;
    }

private:
    svmComponents::MemeticFeaturesEducation& m_educationOfTrainingSet;
};



class MemeticFeaturesAdaptationStrategy
{
public:
    explicit MemeticFeaturesAdaptationStrategy(svmComponents::MemeticFeaturesAdaptation& adaptation)
        :m_adaptation(adaptation)
    {}

    std::tuple<bool, unsigned int> launch(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population)
    {
        m_adaptation.adapt(population);

        return std::make_tuple(m_adaptation.getIsModeLocal(), m_adaptation.getNumberOfClassExamples());
    }

private:
    svmComponents::MemeticFeaturesAdaptation& m_adaptation;
};



class MemeticCompensationGenerationStrategy
{
public:
    explicit MemeticCompensationGenerationStrategy(svmComponents::MemeticFeaturesCompensationGeneration& compensationGenerationAlgorithm)
        : m_compensationGeneration(compensationGenerationAlgorithm)
    {}

    std::string getDescription() const;
    std::vector<unsigned int> launch(
        const std::vector<geneticComponents::Parents<svmComponents::SvmFeatureSetMemeticChromosome>>& parents,
        unsigned int numberOfClassExamples)
    {
        return m_compensationGeneration.generate(parents, numberOfClassExamples);
    }

private:
    svmComponents::MemeticFeaturesCompensationGeneration& m_compensationGeneration;
};



class MemeticCrossoverCompensationStrategy
{
public:
    explicit MemeticCrossoverCompensationStrategy(svmComponents::MemeticFeatureCompensation& crossoverCompensationAlgorithm)
        :m_crossoverCompensation(crossoverCompensationAlgorithm)
    {}

    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> launch(
        geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
        std::vector<unsigned int> compensationInfo)
    {
        m_crossoverCompensation.compensate(population, compensationInfo);
        return population;
    }

private:
    svmComponents::MemeticFeatureCompensation& m_crossoverCompensation;
};


class MemeticSuperIndividualCreationStrategy
{
public:
    explicit MemeticSuperIndividualCreationStrategy(svmComponents::MemeticFeaturesSuperIndividualsGeneration& generationAlgorithm)
        : m_generationAlgorithm(generationAlgorithm)
    {}

    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> launch(unsigned int populationSize,
                                                                                  std::vector<svmComponents::Feature>& supportVectorPool,
                                                                                  unsigned int numberOfClassExamples)
    {
        auto newPopulation = m_generationAlgorithm.createPopulation(populationSize, supportVectorPool, numberOfClassExamples);

        return newPopulation;
    }

private:
    svmComponents::MemeticFeaturesSuperIndividualsGeneration& m_generationAlgorithm;
};



class MemeticUpdateFeaturesPoolStrategy
{
public:
    explicit MemeticUpdateFeaturesPoolStrategy(svmComponents::MemeticFeaturesPool& updateMethod)
        : m_updateMethod(updateMethod)
    {}

    const std::vector<svmComponents::Feature>& launch(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
                                                            const dataset::Dataset<std::vector<float>, float>& trainingSet)
    {
        m_updateMethod.updateFeaturesVectorPool(population, trainingSet);

        return m_updateMethod.getFeaturesPool();
    }

private:
    svmComponents::MemeticFeaturesPool& m_updateMethod;
};
} // namespace svmStrategies
