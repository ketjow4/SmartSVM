
#pragma once

#include <gsl/gsl>

namespace dataset
{
template<typename Sample, typename Label>
class IReadOnlyDataset
{
protected:
    using size_type = typename std::size_t;
public:
    virtual ~IReadOnlyDataset() = default;

    virtual size_type size() const = 0;

    virtual bool empty() const = 0;

    virtual const Label& getLabel(size_type index) const = 0;

    virtual gsl::span<const Label> getLabels() const = 0;

    virtual const Sample& getSample(size_type index) const = 0;

    virtual gsl::span<const Sample> getSamples() const = 0;
};

template<typename Description>
class IReadOnlyDescritpion
{
public:
    virtual ~IReadOnlyDescritpion() = default;

    virtual const Description& getDatasetDescription() const = 0;
};

}// namespace dataset
