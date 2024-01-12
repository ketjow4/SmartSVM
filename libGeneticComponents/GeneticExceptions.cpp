
#include "GeneticExceptions.h"

namespace geneticComponents
{
ValueNotInRange::ValueNotInRange(const std::string& valueName, double value, double min, double max)
    : PlatformException("Value: " + valueName + " is: " + std::to_string(value) + " where it should be in range ("
                        + std::to_string(min) + ", " + std::to_string(max) + ")")
{
}

WrongTruncationCoefficient::WrongTruncationCoefficient(double truncationValue)
    : PlatformException("With truncation coefficient value: " + std::to_string(truncationValue) + " there will be 0 new elements selected")
{
}

UnknownEnumType::UnknownEnumType(const std::string& name, const std::string& enumName)
    : PlatformException("Enum type:" + name + " is unknown name in:" + enumName)
{
}

CrossoverParentsSizeInequality::CrossoverParentsSizeInequality(std::size_t parentASize, std::size_t parentBSize)
    : PlatformException("Parents sizes differ in crossover operator. ParentA: " +
                        std::to_string(parentASize) + " ParentB: " + std::to_string(parentBSize))
{
}

TooSmallChromosomeSize::TooSmallChromosomeSize(unsigned int size, unsigned int minimumSize)
    : PlatformException("Chromosome is too small. Minimum is: " + std::to_string(minimumSize) +
                        " Actual is: " + std::to_string(size))
{
}
} // namespace geneticComponents
