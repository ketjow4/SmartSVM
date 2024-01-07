
#pragma once

#include <filesystem>
#include <string>

#include "libPlatform/PlatformException.h"

namespace phd {namespace data 
{
class UnsupportedFileFormatException final : public platform::PlatformException
{
public:
    explicit UnsupportedFileFormatException(const std::filesystem::path& filepath);
};

class FileNotFoundException final : public platform::PlatformException
{
public:
    explicit FileNotFoundException(const std::string& pathToFile);
};

class WrongClassNumberInDataset final : public platform::PlatformException
{
public:
    explicit WrongClassNumberInDataset(int wrongValue);
};}} // namespace phd {namespace data 