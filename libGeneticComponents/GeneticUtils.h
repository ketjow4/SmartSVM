
#pragma once

#include <vector>

namespace geneticComponents { namespace geneticUtils
{
inline bool allZero(const std::vector<bool>& genes)
{
    return std::all_of(genes.begin(), genes.end(), [](bool gene)
                   {
                       return !gene;
                   });
}
}} // namespace geneticComponents::geneticUtils
