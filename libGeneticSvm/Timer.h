
#pragma once

#include <chrono>

namespace genetic
{
class Timer
{
    using Clock = std::chrono::high_resolution_clock;
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
public:
    Timer();

    void startTimer();
    void addTime(double miliseconds);
    void decreaseTime(double miliseconds);
    std::chrono::milliseconds getTimeMiliseconds() const;
    void pause();
    void contine();

private:
    Time m_begin;
    std::chrono::milliseconds m_counter;
    bool m_isPaused;
};
} // namespace genetic