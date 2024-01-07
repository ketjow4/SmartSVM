
#pragma once

#include <memory>
#include <chrono>
#include "libPlatform/Subtree.h"
#include "MersenneTwister64Rng.h"

namespace random
{
enum class randomNumberGenerator
{
    Unknown,
    Mt_19937
};

const randomNumberGenerator& stringToEnum(const std::string& name);

class RandomNumberGeneratorFactory
{
public:
    static std::unique_ptr<IRandomNumberGenerator> create(const platform::Subtree& config);

private:
    static unsigned long long getSeed(const platform::Subtree& config);
};
} // namespace random