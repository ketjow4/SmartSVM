#pragma once

#include <vector>
#include <algorithm>

inline std::vector<bool> toMask(std::vector<double> info)
{
    std::vector<bool> mask;
    mask.reserve(info.size());

    for(auto f : info)
    {
        mask.emplace_back(f == 0 ? 0 : 1);
    }

    return mask;
}

class SmallerPool
{
private:
    SmallerPool() {};

    std::vector<bool> featureMask;

public:
    static SmallerPool& instance()
    {
        static SmallerPool INSTANCE;
        return INSTANCE;
    }

    void setupMask(std::vector<bool>& mask)
    {
        featureMask = mask;
    }

    size_t size()
    {
        return std::count(featureMask.begin(), featureMask.end(), true);
    }

    bool isOk(uint32_t /*id*/)
    {
        return true;
        //return featureMask[id];
    }
};