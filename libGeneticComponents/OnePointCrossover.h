
#pragma once

#include <memory>
#include <iterator>
#include "LibGeneticComponents/Population.h"
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/BaseCrossoverOperator.h"
#include "LibGeneticComponents/BinaryChromosome.h"

namespace geneticComponents
{
template <class binaryChromosome>
class OnePointCrossover : public BaseCrossoverOperator<binaryChromosome>
{
    static_assert(std::is_base_of<BinaryChromosome, binaryChromosome>::value, "Cannot do binary crossover for class not derived from BinaryChromosome");
public:
    explicit OnePointCrossover(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine);

    Population<binaryChromosome> crossoverParents(const std::vector<Parents<binaryChromosome>>& parents) override;
    binaryChromosome crossoverChromosomes(const binaryChromosome& parentA, const binaryChromosome& parentB) override;

private:
    int getCuttingPoint(std::uniform_int_distribution<int> chromosomeSize);

    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    bool m_isfirstChild = true;
    int m_cuttingPoint;
};

template <class binaryChromosome>
OnePointCrossover<binaryChromosome>::OnePointCrossover(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine)
    : m_rngEngine(std::move(rngEngine))
{
    if(m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

template <class binaryChromosome>
Population<binaryChromosome> OnePointCrossover<binaryChromosome>::crossoverParents(const std::vector<Parents<binaryChromosome>>& parents)
{
    constexpr auto childrenPerPair = 2;
    std::vector<binaryChromosome> children;
    children.reserve(parents.size() * childrenPerPair);

    for(const auto& parentsPair : parents)
    {
        children.emplace_back(this->crossoverChromosomes(parentsPair.first, parentsPair.second));
        children.emplace_back(this->crossoverChromosomes(parentsPair.second, parentsPair.first));
    }

    return geneticComponents::Population<binaryChromosome>(std::move(children));
}

template <class binaryChromosome>
int OnePointCrossover<binaryChromosome>::getCuttingPoint(std::uniform_int_distribution<int> chromosomeSize)
{
    if(m_isfirstChild)
    {
        m_cuttingPoint = m_rngEngine->getRandom(chromosomeSize);
    }
    m_isfirstChild = !m_isfirstChild;
    return m_cuttingPoint;
}

template <class binaryChromosome>
binaryChromosome OnePointCrossover<binaryChromosome>::crossoverChromosomes(const binaryChromosome& parentA,
                                                                           const binaryChromosome& parentB)
{
    if (parentA.getGenes().size() != parentB.getGenes().size())
    {
        throw CrossoverParentsSizeInequality(parentA.getGenes().size(), parentB.getGenes().size());
    }

    std::uniform_int_distribution<int> chromosomeSize(0, static_cast<int>(parentA.getGenes().size()));
    std::vector<bool> child;
    child.reserve(parentA.getGenes().size());

    auto cuttingPoint = getCuttingPoint(chromosomeSize);

    auto parentAGenes = parentA.getGenes();
    auto parentBGenes = parentB.getGenes();

    std::copy_n(parentAGenes.begin(), cuttingPoint, std::back_inserter(child));
    std::copy_n(parentBGenes.begin() + cuttingPoint, parentBGenes.size() - cuttingPoint, std::back_inserter(child));

    binaryChromosome childChromosome(std::move(child));
    return childChromosome;
}
} // namespace geneticComponents
