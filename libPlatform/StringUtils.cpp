

#include <sstream>
#include <iomanip>

#include <string.h>
#include <iostream>                                                              
#include <iomanip>
#include <string>
#include <sstream>
#include <ctime>
#include <vector>
#include <cstring>
#include <cstdio>

#include "StringUtils.h"

namespace platform { namespace stringUtils 
{
std::vector<std::string>& splitString(const std::string& s, char delimiter, std::vector<std::string>& elements)
{
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delimiter))
    {
        elements.push_back(item);
    }
    return elements;
}


// void splitString(std::vector<std::string>& tokens, char* str, const char* delimiters)
// {
//     char* saveptr;
//     char* token;

//     for (token = strtok_s(str, delimiters, &saveptr);
//         token != NULL;
//         token = strtok_s(NULL, delimiters, &saveptr)) {
//         tokens.emplace_back(std::string(token));
//     }
// }

void splitString(std::vector<std::string>& tokens, const char* str, const char* delimiters) {
    std::string s(str);
    std::size_t start = 0;
    std::size_t end = s.find_first_of(delimiters);

    while (end != std::string::npos) {
        if (end > start) {
            tokens.emplace_back(s.substr(start, end - start));
        }
        start = end + 1;
        end = s.find_first_of(delimiters, start);
    }
    if (start < s.length()) {
        tokens.emplace_back(s.substr(start));
    }
}


std::vector<std::string> splitString(const std::string& s, std::string delimiter)
{
    std::vector<std::string> results;
    auto copy = s;
    size_t pos = 0;
    std::string token;
    while ((pos = copy.find(delimiter)) != std::string::npos) {
        token = copy.substr(0, pos);
        results.emplace_back(token);
        copy.erase(0, pos + delimiter.length());
    }

    return results;
}

std::vector<std::string> splitString(const std::string& s, char delimiter)
{
    std::vector<std::string> elems;
    splitString(s, delimiter, elems);
    return elems;
}

template<typename floatingPoint>
std::string toStringWithPrecision(const floatingPoint value, const int n)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(n) << value;
    return out.str();
}

template std::string toStringWithPrecision<>(const float value, const int n);
template std::string toStringWithPrecision<>(const double value, const int n);
}
}
