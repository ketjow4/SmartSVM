
#include <cstdlib> 
#include <fstream>
#include "CsvReader.h"
#include "TabularDataExceptions.h"
#include "libPlatform/StringUtils.h"

//#include "libDataProvider/FastFloat.h"

namespace phd {namespace data 
{
dataset::Dataset<std::vector<float>, float> readCsv(const std::filesystem::path& filename)
{
    dataset::Dataset<std::vector<float>, float> dataset;

    //std::ios::sync_with_stdio(false);
	std::ifstream inFile(filename.native(), std::ios_base::in);
	
    if (inFile.is_open())
    {
        std::string line;
        std::vector<std::string> tokens;
  
        while (std::getline(inFile, line))
        {
			if (line[0] == '@') //case for dataset downloaded from https://sci2s.ugr.es/keel/datasets.php
				continue;

            platform::stringUtils::splitString(line, ',', tokens);
            //platform::stringUtils::splitString(tokens, &line[0], ",");
        	
            std::vector<float> sample;
            sample.reserve(tokens.size());
            
            std::transform(tokens.begin(), --tokens.end(), std::back_inserter(sample), 
                [](const std::string& number)
            {
                 /*   double x = 0.0;
                    fast_double_parser::parse_number(number.c_str(), &x);
                    return static_cast<float>(x);*/
            	
				//return std::strtof(number.c_str(), NULL);
            	
                return std::stof(number);
            });
            
			auto targetValue = std::stof(tokens.back());
			if (targetValue == -1)
			{
				targetValue = 0;
			}
            //const std::vector<float> allowedClassValues = { 0,1 };
            //if(targetValue == allowedClassValues[0] || targetValue == allowedClassValues[1])
            //{
            dataset.addSample(std::move(sample), targetValue);
            /*}
            else
            {
                throw WrongClassNumberInDataset(targetValue);
            }*/
            tokens.clear();
        }
    }
    else
    {
        throw FileNotFoundException(filename.string());
    }
    return dataset;
}

dataset::Dataset<std::vector<float>, float> readCsvGroups(const std::filesystem::path& filename)
{
    dataset::Dataset<std::vector<float>, float> dataset;

    //std::ios::sync_with_stdio(false);
    std::ifstream inFile(filename.native(), std::ios_base::in);

    std::vector<float> groups;

    if (inFile.is_open())
    {
        std::string line;
        std::vector<std::string> tokens;

        while (std::getline(inFile, line))
        {
            if (line[0] == '@') //case for dataset downloaded from https://sci2s.ugr.es/keel/datasets.php
                continue;

            platform::stringUtils::splitString(line, ',', tokens);
            //platform::stringUtils::splitString(tokens, &line[0], ",");

            std::vector<float> sample;
            sample.reserve(tokens.size());


            std::transform(tokens.begin(), ----tokens.end(), std::back_inserter(sample),
                [](const std::string& number)
                {
                    /*   double x = 0.0;
                       fast_double_parser::parse_number(number.c_str(), &x);
                       return static_cast<float>(x);*/

                       //return std::strtof(number.c_str(), NULL);

                    return std::stof(number);
                });

            auto targetValue = std::stof(*(----tokens.end()));
            if (targetValue == -1)
            {
                targetValue = 0;
            }
            //const std::vector<float> allowedClassValues = { 0,1 };
            //if(targetValue == allowedClassValues[0] || targetValue == allowedClassValues[1])
            //{
            dataset.addSample(std::move(sample), targetValue);
            /*}
            else
            {
                throw WrongClassNumberInDataset(targetValue);
            }*/

            //group info
        	auto group = std::stof(*(--tokens.end()));
            groups.emplace_back(group);

            tokens.clear();
        }
        dataset.setGroups(std::move(groups));
    }
    else
    {
        throw FileNotFoundException(filename.string());
    }
    return dataset;
}
}} // namespace phd {namespace data 