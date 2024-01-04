

#include "Percent.h"

namespace platform
{
IncorrectPercentValue::IncorrectPercentValue(double value, double min, double max)
    : PlatformException("Value of percent is: " + std::to_string(value) + " where it should be in range ("
        + std::to_string(min) + ", " + std::to_string(max) + ")")
{
}

Percent::Percent(double value)
    : m_percentValue(value >= m_minPercent && value <= m_maxPercent
                         ? value
                         : throw IncorrectPercentValue(value, m_minPercent, m_maxPercent))
{
}

bool Percent::operator<(const Percent& other) const
{
    return m_percentValue < other.m_percentValue;
}

bool Percent::operator>(const Percent& other) const
{
    return m_percentValue > other.m_percentValue;
}
} // namespace platform
