

#pragma once

#include <exception>
#include <string>
#include "PlatformException.h"

namespace platform
{
class EmptyNameException final : public PlatformException
{
public:
    explicit EmptyNameException()
        : PlatformException("Cannot pass empty name")
    {
    }
};

class EmptyVectorException final : public PlatformException
{
public:
    explicit EmptyVectorException()
        : PlatformException("Cannot pass empty vector")
    {
    }
};

class FileNotFoundException final : public PlatformException
{
public:
    explicit FileNotFoundException(const std::string& pathToFile)
        : PlatformException("Cannot find file in path: " + pathToFile)
    {
    }
};

class JsonParserException final : public PlatformException
{
public:
    explicit JsonParserException(const std::string& errorName)
        : PlatformException("Cannot parse JSON file. Error message: " + errorName)
    {
    }
};

class PropertyNotFoundException final : public PlatformException
{
public:
    explicit PropertyNotFoundException(const std::string& propertyName)
        : PlatformException("Cannot find property: " + propertyName)
    {
    }
};

class ChildNotFoundException final : public PlatformException
{
public:
    explicit ChildNotFoundException(const std::string& childName)
        : PlatformException("Cannot find child named: " + childName)
    {
    }
};

class WrongConversionException final : public PlatformException
{
public:
    explicit WrongConversionException(const std::string& propertyName, const std::string& typeName)
        : PlatformException("Cannot convert property: " + propertyName + " to type: " + typeName)
    {
    }
};

class RootCreationFailedException final : public PlatformException
{
public:
    explicit RootCreationFailedException(const std::string& message)
        : PlatformException(message)
    {
    }
};

class UnhandledBoostException final : public PlatformException
{
public:
    explicit UnhandledBoostException()
        : PlatformException("Unhandled exception coming from Boost library")
    {
    }
};

} // namespace platform
