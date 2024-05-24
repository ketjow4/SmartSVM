
#pragma once

#include <gsl/gsl>
#include "IReadOnlyDataset.h"

namespace dataset
{
template<typename Sample, typename Label>
class IDataset : public IReadOnlyDataset<Sample, Label>
{
public:
    using size_type = typename IReadOnlyDataset<Sample, Label>::size_type;

    virtual ~IDataset() = default;

    virtual void setLabel(size_type index, const Label& label) = 0;

    virtual void setLabel(size_type index, Label&& label) = 0;

    virtual void setLabels(gsl::span<const Label> labels) = 0;

    virtual void setLabels(std::vector<Label>&& labels) = 0;

    virtual void setSample(size_type index, const Sample& sample) = 0;

    virtual void setSamples(gsl::span<const Sample> sample) = 0;

    virtual void setSamples(std::vector<Sample>&& samples) = 0;

    virtual void setSample(size_type index, Sample&& sample) = 0;

    virtual void addSample(const Sample& sample, const Label& label) = 0;

    virtual void addSample(Sample&& sample, Label&& label) = 0;
};

template<typename Description>
class IDescription : public IReadOnlyDescritpion<Description>
{
public:
    virtual void setDatasetDescription(const Description& description) = 0;
};
}// namespace dataset
