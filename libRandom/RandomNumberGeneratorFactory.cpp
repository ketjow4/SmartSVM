
#include <unordered_map>
#include "RandomNumberGeneratorFactory.h"
#include "RandomExceptions.h"

namespace random
{
const randomNumberGenerator& stringToEnum(const std::string& name)
{
    const static std::unordered_map<std::string, randomNumberGenerator> translations =
    {
        { "Mt_19937", randomNumberGenerator::Mt_19937 }
    };

    auto iterator = translations.find(name);
    if (iterator != translations.end())
    {
        return iterator->second;
    }
    throw UnknownRandomNumberGenerator(name);
}

std::unique_ptr<random::IRandomNumberGenerator> RandomNumberGeneratorFactory::create(const platform::Subtree& config)
{
    auto name = config.getValue<std::string>("RandomNumberGenerator.Name");

    switch (stringToEnum(name))
    {
    case randomNumberGenerator::Mt_19937:
        return std::make_unique<random::MersenneTwister64Rng>(getSeed(config));
    default:
        throw UnknownRandomNumberGenerator(name);
    }
}

unsigned long long RandomNumberGeneratorFactory::getSeed(const platform::Subtree& config)
{
    if (config.getValue<bool>("RandomNumberGenerator.IsSeedRandom"))
    {
        return static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    return config.getValue<int>("RandomNumberGenerator.Seed");
}
} // namespace random