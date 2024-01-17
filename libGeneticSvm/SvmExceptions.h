
#pragma once

#include <string>

#include "libPlatform/PlatformException.h"

namespace genetic
{
class NotImplementedException final : public platform::PlatformException
{
public:
    explicit NotImplementedException()
        : PlatformException("Not implemented")
    {
    }
};

class UnknownAlgorithmTypeException final : public platform::PlatformException
{
public:
    explicit UnknownAlgorithmTypeException(const std::string& algorithmName);
};

class UnsupportedAlgorithmTypeException final : public platform::PlatformException
{
public:
    explicit UnsupportedAlgorithmTypeException(const std::string& algorithmName);
};

class ErrorInConfigException final : public platform::PlatformException
{
public:
    explicit ErrorInConfigException(const std::string& message);
};

class BadTrainingElement final : public platform::PlatformException
{
public:
    explicit BadTrainingElement()
        : PlatformException{"Bad training"}
    {
    }
};
} // namespace genetic
