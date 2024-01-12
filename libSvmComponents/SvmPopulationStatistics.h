
#pragma once

#include <set>
#include "libSvmComponents/BaseSvmChromosome.h"
#include "libGeneticComponents/Population.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/SvmFeatureSetChromosome.h"
#include "libSvmComponents/SvmFeatureSetMemeticChromosome.h"
#include "SvmSimultaneousChromosome.h"
#include "SvmCustomKernelChromosome.h"
#include "SvmCustomKernelFeaturesSelectionChromosome.h"

namespace svmComponents
{
template <typename chromosome>
class SvmPopulationStatistics
{
public:
    static double getMeanNumberOfSupportVectors(const geneticComponents::Population<chromosome>& population);

    static unsigned int calculateUniqueElements(const geneticComponents::Population<chromosome>& population);

    static double getMeanTimeOfClassification(const geneticComponents::Population<chromosome>& population);
};

template <typename chromosome>
double SvmPopulationStatistics<chromosome>::getMeanNumberOfSupportVectors(const geneticComponents::Population<chromosome>& population)
{
    static_assert(std::is_base_of<BaseSvmChromosome, chromosome>::value, "T must be derived from BaseSvmChromosome class");

    if (population.empty())
    {
        throw geneticComponents::PopulationIsEmptyException();
    }

    double sumOfSupportVectors = std::accumulate(population.begin(), population.end(), 0.0,
                                                 [](double sum, const auto& individual)
                                             {
                                                 return sum + individual.getNumberOfSupportVectors();
                                             });
    return sumOfSupportVectors / population.size();
}

template <typename chromosome>
struct comparator
{
    bool operator()(const chromosome& lhs, const chromosome& rhs) const;
};

template <>
inline bool comparator<SvmTrainingSetChromosome>::operator()(const SvmTrainingSetChromosome& lhs, const SvmTrainingSetChromosome& rhs) const
{
    return lhs.getDataset() < rhs.getDataset();
}

template <>
inline bool comparator<SvmFeatureSetChromosome>::operator()(const SvmFeatureSetChromosome& lhs, const SvmFeatureSetChromosome& rhs) const
{
    return lhs.getGenes() < rhs.getGenes();
}

template <>
inline bool comparator<SvmKernelChromosome>::operator()(const SvmKernelChromosome& lhs, const SvmKernelChromosome& rhs) const
{
    return lhs.getKernelParameters() < rhs.getKernelParameters();
}

template <>
inline bool comparator<SvmFeatureSetMemeticChromosome>::operator()(const SvmFeatureSetMemeticChromosome& lhs, const SvmFeatureSetMemeticChromosome& rhs) const
{
    return lhs.getDataset() < rhs.getDataset();
}

template <>
inline bool comparator<SvmSimultaneousChromosome>::operator()(const SvmSimultaneousChromosome& lhs, const SvmSimultaneousChromosome& rhs) const
{
    return lhs.getDataset() < rhs.getDataset() && 
           lhs.getKernelParameters() < rhs.getKernelParameters() && 
           lhs.getFeatures() < rhs.getFeatures();
}

template <>
inline bool comparator<SvmCustomKernelChromosome>::operator()(const SvmCustomKernelChromosome& lhs, const SvmCustomKernelChromosome& rhs) const
{
    return lhs.getDataset() < rhs.getDataset();
}

template <>
inline bool comparator<BaseSvmChromosome>::operator()(const BaseSvmChromosome& lhs, const BaseSvmChromosome& rhs) const
{
    return lhs.getNumberOfSupportVectors() < rhs.getNumberOfSupportVectors(); //this does not make sense, only for quick experiments in order to fix code for EnsembleTree logger
}


template <>
inline bool comparator<SvmCustomKernelFeaturesSelectionChromosome>::operator()(const SvmCustomKernelFeaturesSelectionChromosome& lhs, const SvmCustomKernelFeaturesSelectionChromosome& rhs) const
{
    return lhs.getFeatures() < rhs.getFeatures() &&
        lhs.getKernelDataset() < rhs.getKernelDataset();
}

template <typename chromosome>
unsigned SvmPopulationStatistics<chromosome>::calculateUniqueElements(const geneticComponents::Population<chromosome>& population)
{
    std::set<chromosome, comparator<chromosome>> helper;
    for (auto i = 0u; i < population.size(); i++)
    {
        helper.insert(population[i]);
    }
    if (helper.size() == population.size())
    {
        return static_cast<unsigned int>(helper.size());
    }
    // @wdudzik if not all are inserted that means at least 2 are identical and 1 of them is not inserted
    // that why there is substract 1
    return static_cast<unsigned int>(helper.size() - 1);
}

template <typename chromosome>
double SvmPopulationStatistics<chromosome>::getMeanTimeOfClassification(const geneticComponents::Population<chromosome>& population)
{
    static_assert(std::is_base_of<BaseSvmChromosome, chromosome>::value, "T must be derived from BaseSvmChromosome class");

    if (population.empty())
    {
        throw geneticComponents::PopulationIsEmptyException();
    }

    double sumOfClassificationTime = std::accumulate(population.begin(), population.end(), 0.0,
                                                     [](double sum, const auto& individual)
                                                 {
                                                     return sum + individual.getTime().count();
                                                 });
    return sumOfClassificationTime / population.size();
}
} // namespace svmComponents
