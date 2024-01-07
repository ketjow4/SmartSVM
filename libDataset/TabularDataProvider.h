
#pragma once


#include "CsvReader.h"
#include "TabularDataExceptions.h"


namespace phd {namespace data 
{
class TabularDataProvider
{
public:
    dataset::Dataset<std::vector<float>, float> loadData(const std::filesystem::path& pathToFile) const;
    
};

inline dataset::Dataset<std::vector<float>, float> TabularDataProvider::loadData(const std::filesystem::path& pathToFile) const
{
	auto result = pathToFile.extension().string();

    if (result == ".csv")
    {
        return readCsv(pathToFile);
    }
    if (result == ".groups")
    {
        return readCsvGroups(pathToFile);
    }
    throw UnsupportedFileFormatException(pathToFile);
}
}} // namespace phd {namespace data 