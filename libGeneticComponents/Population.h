
#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include "LibGeneticComponents/BaseChromosome.h"
#include "LibGeneticComponents/IOperator.h"
#include "LibGeneticComponents/GeneticExceptions.h"

namespace geneticComponents
{
template <typename T>
class Population
{
    static_assert(std::is_base_of<BaseChromosome, T>::value, "T must be derived from BaseChromosome class");
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
public:

    Population();
    explicit Population(std::vector<T> population);

    const T& getBestOne() const;
    double getMeanFitness() const;
    unsigned int getBestIndividualIndex() const;
    
    const T& operator[](int idx) const;
    T& operator[](int idx);
    size_t size() const;
    bool empty() const;
    std::vector<T>& get();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    void swap(Population<T>& right) noexcept;

    void applyOperator(IOperator<T>& op);

private:
    std::vector<T> m_population;
};

template <typename T>
Population<T>::Population()
{
}

template <typename T>
Population<T>::Population(std::vector<T> population) : m_population(std::move(population))
{
}

template <typename T>
const T& Population<T>::getBestOne() const
{
    if(m_population.empty())
    {
        throw PopulationIsEmptyException();
    }
    return *std::max_element(m_population.begin(), m_population.end());
}

template <typename T>
double Population<T>::getMeanFitness() const
{
    if (m_population.empty())
    {
        throw PopulationIsEmptyException();
    }
    double sumFitness = std::accumulate(m_population.begin(), m_population.end(), 0.0,
                                        [](double sum, const auto& chromosome)
                                        {
                                            return sum + chromosome.getFitness();
                                        });
    return sumFitness / m_population.size();
}

template <typename T>
unsigned int Population<T>::getBestIndividualIndex() const
{
    if (m_population.empty())
    {
        throw PopulationIsEmptyException();
    }
    auto bestIterator = std::max_element(m_population.begin(), m_population.end());
    return static_cast<unsigned int>(std::distance(m_population.begin(), bestIterator));
}

template <typename T>
void Population<T>::applyOperator(IOperator<T>& op)
{
    op(*this);
}

template <typename T>
std::vector<T>& Population<T>::get()
{
    return m_population;
}

template <typename T>
typename Population<T>::iterator Population<T>::begin()
{
    return m_population.begin();
}

template <typename T>
typename Population<T>::iterator Population<T>::end()
{
    return m_population.end();
}

template <typename T>
typename Population<T>::const_iterator Population<T>::begin() const
{
    return  m_population.cbegin();
}

template <typename T>
typename Population<T>::const_iterator Population<T>::end() const
{
    return  m_population.cend();
}

template <typename T>
void Population<T>::swap(Population<T>& right) noexcept
{
    m_population.swap(right.get());
}

template <typename T>
const T& Population<T>::operator[](int idx) const
{
    return m_population[idx];
}

template <typename T>
T& Population<T>::operator[](int idx)
{
    return m_population[idx];
}

template <typename T>
size_t Population<T>::size() const
{
    return m_population.size();
}

template <typename T>
bool Population<T>::empty() const
{
    return m_population.empty();
}
} // namespace geneticComponents
