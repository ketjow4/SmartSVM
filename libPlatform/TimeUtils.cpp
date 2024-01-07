#include <iomanip>
#include <chrono>
#include <sstream>

#include "TimeUtils.h"

namespace timeUtils
{  
std::tm getLocalTime()
{
    constexpr auto beginYearInTmStruct = 1900;
    // @wdudzik tm_mon in local time struct is time since January so it starts from 0 to 11. In calendar we count month from 1 to 12
    // so to have calendar date format we have to add startMonth = 1.
    constexpr auto startMonth = 1;  
    auto timePoint = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::to_time_t(timePoint);
    std::tm localTime;
    localtime_s(&localTime, &now);

    localTime.tm_year += beginYearInTmStruct;
    localTime.tm_mon += startMonth;

    return localTime;
}

std::tm makeTmStruct(int year, int month, int day)
{
    constexpr auto beginYearInTmStruct = 1900;
    constexpr auto startMonth = 1;

    std::tm tm = {0};
    tm.tm_year = year - beginYearInTmStruct;
    tm.tm_mon = month - startMonth; 
    tm.tm_mday = day; 

    return tm;
}


std::string getTimestamp()
{
    auto time = timeUtils::getLocalTime();

    using namespace std::chrono;
    auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    std::stringstream ss;

    ss << time.tm_year << '-'
        << std::setfill('0') << std::setw(2) << time.tm_mon << '-'
        << std::setfill('0') << std::setw(2) << time.tm_mday << '-'
        << std::setfill('0') << std::setw(2) << time.tm_hour << '_'
        << std::setfill('0') << std::setw(2) << time.tm_min << '_'
        << std::setfill('0') << std::setw(2) << time.tm_sec << '.'
        << std::setfill('0') << std::setw(3) << std::to_string(ms.count() % 1000) << '_';

    return ss.str();
}

std::string getShortTimestamp()
{
    auto time = timeUtils::getLocalTime();

    using namespace std::chrono;
    auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    std::stringstream ss;

    ss  << std::setfill('0') << std::setw(2) << time.tm_hour << '_'
        << std::setfill('0') << std::setw(2) << time.tm_min << '_'
        << std::setfill('0') << std::setw(2) << time.tm_sec << '.'
        << std::setfill('0') << std::setw(3) << std::to_string(ms.count() % 1000) << '_';

    return ss.str();
}
} // namespace timeUtils