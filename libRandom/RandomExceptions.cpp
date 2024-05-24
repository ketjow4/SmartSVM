
#include "RandomExceptions.h"

namespace my_random
{
UnknownRandomNumberGenerator::UnknownRandomNumberGenerator(const std::string& name)
    : PlatformException("Random nubmer generator: " + name + " is unknown")
{
}
} // namespace random