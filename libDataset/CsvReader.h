
#pragma once

#include <vector>

#include <filesystem>
#include "Dataset.h"

namespace phd {namespace data 
{
/***
 * Supported format should look like:
 * 
 * 1,2,0
 * 3,4,1
 * 5,6,0
 * 
 * Where the last column is treated as class value (int) and rest values are features.
 * All features should be numbers convertable to float.
 * 
 * Line endings should be platform specific:
 *  - Windows (\r\n or \n)
 *  - Linux (\n)  - not tested (as state in documentation)
 *  - MacOS (\n)  - not tested (as state in documentation)
 *  
 *  If line endings are not correct behaviour is undefined.
 */
dataset::Dataset<std::vector<float>, float> readCsv(const std::filesystem::path& filename);

/***
 * Supported format should look like:
 *
 * 2 features, class, group
 * 1,2,			1,		1
 * 3,4,			1,		2
 * 5,6,			0,		2
 *
 * Where the last column is treated as group value (int) and last but one is class value and rest of the values are features.
 * All features should be numbers convertable to float.
 *
 * Line endings should be platform specific:
 *  - Windows (\r\n or \n)
 *  - Linux (\n)  - not tested (as state in documentation)
 *  - MacOS (\n)  - not tested (as state in documentation)
 *
 *  If line endings are not correct behaviour is undefined.
 */
dataset::Dataset<std::vector<float>, float> readCsvGroups(const std::filesystem::path& filename);
}} // namespace phd {namespace data 
