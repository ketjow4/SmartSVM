

#include "PlatformException.h"

namespace platform
{
PlatformException::PlatformException(std::string message)
    : m_message{std::move(message)}
{
}

char const* PlatformException::what() const
{
    return m_message.c_str();
}
} // namespace platform
