
#include "RandomExceptions.h"

namespace random
{
UnknownRandomNumberGenerator::UnknownRandomNumberGenerator(const std::string& name)
    : PlatformException("Random nubmer generator: " + name + " is unknown")
{
}
} // namespace random