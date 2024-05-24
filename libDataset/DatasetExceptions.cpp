
#include "DatasetExceptions.h"

namespace dataset
{
DifferentSizeException::DifferentSizeException(std::size_t newDataSize, std::size_t oldDataSize)
    : PlatformException("Sizes are not the same. New data size is: "
        + std::to_string(newDataSize)
        + ", while old data size is: "
        + std::to_string(oldDataSize))
{
}

OutOfBoundsException::OutOfBoundsException(std::size_t index, std::size_t datasetSize)
    : PlatformException("Subscript out of bounds. Index used: "
        + std::to_string(index)
        + ", while dataset size is: "
        + std::to_string(datasetSize))
{
}
} // namespace platform