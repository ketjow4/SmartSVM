
#include "TabularDataExceptions.h"

namespace phd {namespace data 
{
UnsupportedFileFormatException::UnsupportedFileFormatException(const std::filesystem::path& filepath)
    : PlatformException("File: " + filepath.string() + " have unsupported format")
{
}

FileNotFoundException::FileNotFoundException(const std::string& pathToFile)
    : PlatformException("Cannot find file in path: " + pathToFile)
{
}

WrongClassNumberInDataset::WrongClassNumberInDataset(int wrongValue)
    : PlatformException("Founded class value " + std::to_string(wrongValue) + " where it should be 0 or 1. Only 2 class classification is supported.")
{
}
}} // namespace phd {namespace data 