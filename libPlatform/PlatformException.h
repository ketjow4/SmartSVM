#pragma once

#include <string>

namespace platform
{
class PlatformException : public std::exception
{
public:
    char const* what() const override final;

protected:
    explicit PlatformException(std::string message);

private:
    const std::string m_message;
};
} // namespace platform
