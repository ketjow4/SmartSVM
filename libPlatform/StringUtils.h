

#pragma once

#include <string>
#include <vector>

namespace platform { namespace stringUtils
{
    std::vector<std::string>& splitString(const std::string& s, char delimiter, std::vector<std::string>& elements);

    std::vector<std::string> splitString(const std::string& s, char delimiter);

    std::vector<std::string> splitString(const std::string& s, std::string delimiter);

    void splitString(std::vector<std::string>& tokens, char* str, const char* delimiters);

    template<typename floatingPoint>
    std::string toStringWithPrecision(const floatingPoint value, const int n = 2);
}
}
