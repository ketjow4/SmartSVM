#pragma once

#include <chrono>
#include <string>
#include <ctime>

namespace timeUtils
{
std::tm getLocalTime();

std::tm makeTmStruct(int year, int month, int day);

std::string getTimestamp();

std::string getShortTimestamp();
} // namespace timeUtils