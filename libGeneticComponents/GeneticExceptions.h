
#pragma once

#include <string>
#include "libPlatform/PlatformException.h"

namespace geneticComponents
{
class ValueNotInRange final : public platform::PlatformException
{
public:
    explicit ValueNotInRange(const std::string& valueName, double value, double min, double max);
};

class PopulationIsEmptyException final : public platform::PlatformException
{
public:
    explicit PopulationIsEmptyException()
        : PlatformException{"Population is empty"}
    {
    }
};

class WrongTruncationCoefficient final : public platform::PlatformException
{
public:
    explicit WrongTruncationCoefficient(double truncationValue);
};

class UnknownEnumType final : public platform::PlatformException
{
public:
    explicit UnknownEnumType(const std::string& name, const std::string& enumName);
};

class RandomNumberGeneratorNullPointer final : public platform::PlatformException
{
public:
    explicit RandomNumberGeneratorNullPointer()
        : PlatformException{"Random number generator null pointer recevied"}
    {
    }
};

class CrossoverParentsSizeInequality final : public platform::PlatformException
{
public:
    explicit CrossoverParentsSizeInequality(std::size_t parentASize, std::size_t parentBSize);
};

class TooSmallChromosomeSize final : public platform::PlatformException
{
public:
    TooSmallChromosomeSize(unsigned int size, unsigned int minimumSize);
};
} // namespace geneticComponents
