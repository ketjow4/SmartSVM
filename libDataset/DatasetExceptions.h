
#pragma once

#include <string>
#include "libPlatform/PlatformException.h"

namespace dataset
{
class DifferentSizeException final : public platform::PlatformException
{
public:
    explicit DifferentSizeException(std::size_t newDataSize, std::size_t oldDataSize);
};

class OutOfBoundsException final : public platform::PlatformException
{
public:
    explicit OutOfBoundsException(std::size_t index, std::size_t datasetSize);
};

class DescriptionIsNullInDatasetView final : public platform::PlatformException
{
public:
    explicit DescriptionIsNullInDatasetView()
        : PlatformException("Description in dataset view is a nullptr")
    {
    }
};
} // namespace platform
