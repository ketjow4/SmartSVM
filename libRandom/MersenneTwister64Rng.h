
#pragma once

#include <random>
#include "IRandomNumberGenerator.h"

namespace my_random
{
class MersenneTwister64Rng : public IRandomNumberGenerator
{
public:
    explicit MersenneTwister64Rng(uint64_t seed);

    double getRandom(std::uniform_real_distribution<double>& distribution) override;
    int getRandom(std::uniform_int_distribution<int>& distribution) override;
    bool getRandom(std::bernoulli_distribution& distribution) override;

private:
     std::mt19937_64 m_engine;
};


inline double MersenneTwister64Rng::getRandom(std::uniform_real_distribution<double>& distribution)
{
    return distribution(m_engine);
}

inline int MersenneTwister64Rng::getRandom(std::uniform_int_distribution<int>& distribution)
{
    return distribution(m_engine);
}

inline bool MersenneTwister64Rng::getRandom(std::bernoulli_distribution& distribution)
{
    return distribution(m_engine);
}
} // namespace my_random

