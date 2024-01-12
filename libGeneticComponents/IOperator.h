
#pragma once

namespace geneticComponents
{
template<typename T>
class Population;

template<typename T>
class IOperator
{
public:
    virtual ~IOperator() = default;
    
    virtual void operator()(Population<T>& population) = 0;
};
} // namespace geneticComponents
