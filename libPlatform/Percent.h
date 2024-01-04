

#pragma once
#include "PlatformException.h"

namespace platform
{
class IncorrectPercentValue final : public PlatformException
{
public:
    explicit IncorrectPercentValue(double value, double min, double max);
};

class Percent
{
public:
    explicit Percent(double value);
    
    bool operator<(const Percent& other) const;
    bool operator>(const Percent& other) const;

    const double m_percentValue;

    static constexpr double m_minPercent = 0.0;
    static constexpr double m_maxPercent = 1.0;
};
} // namespace platform
