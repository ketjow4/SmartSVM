
#pragma once

#include <exception>
#include <string>
#include "libPlatform/PlatformException.h"

namespace my_random
{
class UnknownRandomNumberGenerator final : public platform::PlatformException
{
public:
    explicit UnknownRandomNumberGenerator(const std::string& name);
};
} // namespace random