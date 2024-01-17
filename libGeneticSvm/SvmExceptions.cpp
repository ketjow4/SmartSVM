
#include "SvmExceptions.h"

namespace genetic
{
UnknownAlgorithmTypeException::UnknownAlgorithmTypeException(const std::string& algorithmName)
    : PlatformException("Algorithm: " + algorithmName + " is unknown. Check configuration")
{
}

UnsupportedAlgorithmTypeException::UnsupportedAlgorithmTypeException(const std::string& algorithmName)
    : PlatformException("Algorithm: " + algorithmName + " is currently unsupported")
{
}

ErrorInConfigException::ErrorInConfigException(const std::string& message)
    : PlatformException("Error occured in provided config: " + message)
{
}
} // namespace genetic
