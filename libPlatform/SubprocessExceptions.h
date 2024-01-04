#pragma once

#include "PlatformException.h"

namespace platform::subprocess
{
class CouldNotCreateProcess final : public ::platform::PlatformException
{
public:
    explicit CouldNotCreateProcess(const std::string& message, const std::string& command)
        : PlatformException("Process couldn't be created. Message: " + message + " Command: " + command)
    {
    }
};

class ErrorOnCreatingPipes final : public ::platform::PlatformException
{
public:
    explicit ErrorOnCreatingPipes(const std::string& message)
        : PlatformException("Pipes couldn't be created or configured  properly. Message: " + message)
    {
    }
};

class CouldNotRetriveExitCode final : public ::platform::PlatformException
{
public:
    explicit CouldNotRetriveExitCode(const std::string& message)
        : PlatformException("Could not retrive exit code from command. Message: " + message)
    {
    }
};


} // namespace platform::subprocess
