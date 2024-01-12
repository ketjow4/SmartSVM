
#pragma once

#include "libDataset/Dataset.h"
#include "libSvmComponents/SvmConfigStructures.h"

namespace svmComponents
{

class FoldCreator
{
public:
    explicit FoldCreator(unsigned int numberOfFolds, const dataset::Dataset<std::vector<float>, float>& dataset);

    std::pair<dataset::Dataset<std::vector<float>, float>, dataset::Dataset<std::vector<float>, float>> getFold(unsigned foldNumber);

private:
    unsigned int m_numberOfFolds;
    const dataset::Dataset<std::vector<float>, float>& m_dataset;
};
} // namespace svmComponents