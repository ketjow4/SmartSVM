

#pragma once

#include <string>

#include "libPlatform/PlatformException.h"

namespace strategies
{
class FeatureNotSupportedException final : public platform::PlatformException
{
public:
    explicit FeatureNotSupportedException(const std::string& featureName)
        : PlatformException("The requested feature " + featureName + " is currently unsupported")
    {
    }
};
} // namespace strategies
