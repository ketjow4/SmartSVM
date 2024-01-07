
#pragma once
#include <random>

namespace random
{
class IRandomNumberGenerator
{
public:
    virtual ~IRandomNumberGenerator() = default;
    
    virtual double getRandom(std::uniform_real_distribution<double>& distribution) = 0;
    
    virtual int getRandom(std::uniform_int_distribution<int>& distribution) = 0;

    virtual bool getRandom(std::bernoulli_distribution& distribution) = 0;
};
} // namespace random
