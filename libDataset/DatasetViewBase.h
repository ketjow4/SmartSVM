
#pragma once

#include "IReadOnlyDataset.h"
#include "DatasetExceptions.h"

namespace dataset
{
template <typename Sample, typename Label>
class DatasetViewBase : public IReadOnlyDataset<Sample, Label>
{
public:
    using typename IReadOnlyDataset<Sample, Label>::size_type;

    size_type size() const override;

    const Label& getLabel(size_type index) const override;

    gsl::span<const Label> getLabels() const override;

    const Sample& getSample(size_type index) const override;

    gsl::span<const Sample> getSamples() const override;

    bool empty() const override;

protected:
    DatasetViewBase(const DatasetBase<Sample, Label>& dataset);

    DatasetViewBase() = default;

    DatasetViewBase(DatasetViewBase&& dataset) = default;

    DatasetViewBase& operator=(DatasetViewBase&& dataset) = default;

    DatasetViewBase(const DatasetViewBase&) = default;

    DatasetViewBase& operator=(const DatasetViewBase&) = default;

    ~DatasetViewBase() = default;

private:
    gsl::span<const Sample> m_samples;
    gsl::span<const Label> m_labels;
};

template <typename Sample, typename Label>
DatasetViewBase<Sample, Label>::DatasetViewBase(const DatasetBase<Sample, Label>& dataset)
    : m_samples(dataset.getSamples())
    , m_labels(dataset.getLabels())
{
}

template <typename Sample, typename Label>
typename DatasetViewBase<Sample, Label>::size_type DatasetViewBase<Sample, Label>::size() const
{
    return m_samples.size();
}

template <typename Sample, typename Label>
const Label& DatasetViewBase<Sample, Label>::getLabel(size_type index) const
{
    if (index < size())
    {
        return m_labels[index];
    }
    throw OutOfBoundsException(index, size());
}

template <typename Sample, typename Label>
gsl::span<const Label> DatasetViewBase<Sample, Label>::getLabels() const
{
    return m_labels;
}

template <typename Sample, typename Label>
const Sample& DatasetViewBase<Sample, Label>::getSample(size_type index) const
{
    if (index < size())
    {
        return m_samples[index];
    }
    throw OutOfBoundsException(index, size());
}

template <typename Sample, typename Label>
gsl::span<const Sample> DatasetViewBase<Sample, Label>::getSamples() const
{
    return m_samples;
}

template <typename Sample, typename Label>
bool DatasetViewBase<Sample, Label>::empty() const
{
    return size() == 0;
}
} //namespace dataset
