
#pragma once

#include <iterator>
#include "LibGeneticComponents/ISelectionOperator.h"
#include "libPlatform/Percent.h"

namespace geneticComponents
{
template<typename T>
class TruncationSelection : public ISelectionOperator<T>
{
public:
    explicit TruncationSelection(platform::Percent m_truncationCoefficient);

    Population<T> selectNextGeneration(Population<T>& currentGeneration) override;
    void validatePopulation(Population<T>& currentGeneration) const;

private:
    platform::Percent m_truncationCoefficient;
};

template <typename T>
TruncationSelection<T>::TruncationSelection(platform::Percent truncationCoefficient) : m_truncationCoefficient(truncationCoefficient)
{
}

template <typename T>
Population<T> TruncationSelection<T>::selectNextGeneration(Population<T>& currentGeneration)
{
    validatePopulation(currentGeneration);
    const auto newPopulationSize = static_cast<int>(currentGeneration.size() * m_truncationCoefficient.m_percentValue);
    std::vector<T> nextGeneration;
    nextGeneration.reserve(newPopulationSize);

    std::sort(currentGeneration.begin(), currentGeneration.end());
    std::copy_n(currentGeneration.begin() + (currentGeneration.size() - newPopulationSize), newPopulationSize, std::back_inserter(nextGeneration));

    return Population<T>(std::move(nextGeneration));
}

template <typename T>
void TruncationSelection<T>::validatePopulation(Population<T>& currentGeneration) const
{
    if(currentGeneration.empty())
    {
        throw PopulationIsEmptyException();
    }
    if((static_cast<int>(currentGeneration.size() * m_truncationCoefficient.m_percentValue)) == 0)
    {
        throw WrongTruncationCoefficient(m_truncationCoefficient.m_percentValue);
    }
}



template<typename T>
class ConstantTruncationSelection : public ISelectionOperator<T>
{
public:
    explicit ConstantTruncationSelection(int newSize);

    Population<T> selectNextGeneration(Population<T>& currentGeneration) override;
    void validatePopulation(Population<T>& currentGeneration) const;

    void setNewPopulationSize(int newSize);

private:
    int m_newPopulationSize;
};

template <typename T>
ConstantTruncationSelection<T>::ConstantTruncationSelection(int newSize) : m_newPopulationSize(newSize)
{
}

template <typename T>
Population<T> ConstantTruncationSelection<T>::selectNextGeneration(Population<T>& currentGeneration)
{
    validatePopulation(currentGeneration);
    std::vector<T> nextGeneration;
    nextGeneration.reserve(m_newPopulationSize);

    std::sort(currentGeneration.begin(), currentGeneration.end());
    std::copy_n(currentGeneration.begin() + (currentGeneration.size() - m_newPopulationSize), m_newPopulationSize, std::back_inserter(nextGeneration));

    return Population<T>(std::move(nextGeneration));
}

template <typename T>
void ConstantTruncationSelection<T>::validatePopulation(Population<T>& currentGeneration) const
{
    if (currentGeneration.empty())
    {
        throw PopulationIsEmptyException();
    }
}

template <typename T>
void ConstantTruncationSelection<T>::setNewPopulationSize(int newSize)
{
    m_newPopulationSize = newSize;
}
} // namespace geneticComponents
