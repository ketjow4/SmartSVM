
#pragma once

namespace geneticComponents
{
class BaseChromosome
{
public:
    virtual ~BaseChromosome() = default;
    BaseChromosome();

    bool operator<(const BaseChromosome &rhs) const;

    virtual void updateFitness(double fitness);

    double getFitness() const;

protected:
    double m_fitness;
};

inline bool BaseChromosome::operator<(const BaseChromosome& rhs) const
{
    return m_fitness < rhs.m_fitness;
}

inline void BaseChromosome::updateFitness(double fitness)
{
    m_fitness = fitness;
}

inline double BaseChromosome::getFitness() const
{
    return m_fitness;
}
} // namespace geneticComponents