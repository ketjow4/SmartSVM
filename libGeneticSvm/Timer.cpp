

#include "Timer.h"

namespace genetic
{
Timer::Timer()
{
    m_begin = Clock::now();
    m_isPaused = false;
    m_counter = std::chrono::milliseconds(0);
}

void Timer::startTimer()
{
    m_begin = Clock::now();
}

void Timer::addTime(double miliseconds)
{
    long long temp = static_cast<long long>(miliseconds);
    m_counter += std::chrono::milliseconds(temp);
}

void Timer::decreaseTime(double miliseconds)
{
    long long temp = static_cast<long long>(miliseconds);
    m_counter -= std::chrono::milliseconds(temp);
}

std::chrono::milliseconds Timer::getTimeMiliseconds() const
{
    if(m_isPaused)
    {
        return m_counter;
    }
    else
    {
        return m_counter + std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - m_begin);
    }
}

void Timer::pause()
{
    if(!m_isPaused)
    {
        auto now = Clock::now();
        m_counter += std::chrono::duration_cast<std::chrono::milliseconds>(now - m_begin);
        m_isPaused = true;
    }
}

void Timer::contine()
{
    if(m_isPaused)
    {
        m_begin = Clock::now();
        m_isPaused = false;
    }
}
} // namespace genetic