
#pragma once
#include "libDataset/Dataset.h"

namespace genetic
{
class IDatasetLoader
{
public:
    virtual ~IDatasetLoader() = default;

    virtual const dataset::Dataset<std::vector<float>, float>& getTraningSet() = 0;
    virtual const dataset::Dataset<std::vector<float>, float>& getValidationSet() = 0;
    virtual const dataset::Dataset<std::vector<float>, float>& getTestSet() = 0;
    virtual bool isDataLoaded() const = 0;
    virtual const std::vector<float>& scalingVectorMin() = 0;
    virtual const std::vector<float>& scalingVectorMax() = 0;
};
} // namespace genetic
