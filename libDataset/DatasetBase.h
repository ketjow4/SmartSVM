
#pragma once

#include <vector>
#include <gsl/gsl>
#include "IDataset.h"
#include "DatasetExceptions.h"

namespace dataset
{
#pragma warning( push )
#pragma warning( disable : 4505) // @wdudzik Warning C4505 unreferenced local function has been removed, this happen when not all functions are used

template <typename Sample, typename Label>
class DatasetBase : public IDataset<Sample, Label>
{
public:
    using size_type = typename IReadOnlyDataset<Sample, Label>::size_type;

    static_assert(sizeof(size_type) == sizeof(typename std::vector<Sample>::size_type), "Size type for std::vector<Sample> is different than std::size_t");

    static_assert(sizeof(size_type) == sizeof(typename std::vector<Label>::size_type), "Size type for std::vector<Label> is different than std::size_t");

    size_type size() const override;

    const Label& getLabel(size_type index) const override;

    gsl::span<const Label> getLabels() const override;

    void setLabel(size_type index, const Label& label) override;

    void setLabel(size_type index, Label&& label) override;

    void setLabels(gsl::span<const Label> labels) override;

    void setLabels(std::vector<Label>&& label) override;

    const Sample& getSample(size_type index) const override;

    gsl::span<const Sample> getSamples() const override;

    void setSample(size_type index, const Sample& sample) override;

    void setSample(size_type index, Sample&& label) override;

    void setSamples(gsl::span<const Sample> samples) override;

    void setSamples(std::vector<Sample>&& samples) override;

    void addSample(const Sample& sample, const Label& label) override;

    void addSample(Sample&& sample, Label&& label) override;

    bool empty() const override;

    void reserve(size_t size);

//protected:
    DatasetBase() = default;

    DatasetBase(DatasetBase&& dataset) = default;

    DatasetBase& operator=(DatasetBase&& dataset) = default;

    DatasetBase(const gsl::span<const Sample> samples, const gsl::span<const Label> labels);

    DatasetBase(std::vector<Sample>&& samples, std::vector<Label>&& labels);

    DatasetBase(const DatasetBase&) = default;

    DatasetBase& operator=(const DatasetBase&) = default;

    ~DatasetBase() = default;

private:
    std::vector<Sample> m_samples;
    std::vector<Label> m_labels;
};

template <typename Sample, typename Label>
typename DatasetBase<Sample, Label>::size_type DatasetBase<Sample, Label>::size() const
{
    return m_samples.size();
}

template <typename Sample, typename Label>
const Label& DatasetBase<Sample, Label>::getLabel(size_type index) const
{
    if (index < size())
    {
        return m_labels[index];
    }
    throw OutOfBoundsException(index, size());
}

template <typename Sample, typename Label>
gsl::span<const Label> DatasetBase<Sample, Label>::getLabels() const
{
    return m_labels;
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setLabel(size_type index, const Label& label)
{
    if (index < size())
    {
        m_labels[index] = label;
    }
    else
    {
        throw OutOfBoundsException(index, size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setLabel(size_type index, Label&& label)
{
    if (index < size())
    {
        m_labels[index] = std::move(label);
    }
    else
    {
        throw OutOfBoundsException(index, size());
    }
}

template <typename Sample, typename Label>
const Sample& DatasetBase<Sample, Label>::getSample(size_type index) const
{
    if (index < size())
    {
        return m_samples[index];
    }
    throw OutOfBoundsException(index, size());
}

template <typename Sample, typename Label>
gsl::span<const Sample> DatasetBase<Sample, Label>::getSamples() const
{
    return m_samples;
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setSample(size_type index, const Sample& sample)
{
    if (index < size())
    {
        m_samples[index] = sample;
    }
    else
    {
        throw OutOfBoundsException(index, size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setSample(size_type index, Sample&& sample)
{
    if (index < size())
    {
        m_samples[index] = std::move(sample);
    }
    else
    {
        throw OutOfBoundsException(index, size());
    }
}

template <typename Sample, typename Label>
DatasetBase<Sample, Label>::DatasetBase(const gsl::span<const Sample> samples, const gsl::span<const Label> labels)
{
    if (samples.size() == labels.size())
    {
        m_samples.assign(samples.begin(), samples.end());
        m_labels.assign(labels.begin(), labels.end());
    }
    else
    {
        throw DifferentSizeException(samples.size(), labels.size());
    }
}

template <typename Sample, typename Label>
DatasetBase<Sample, Label>::DatasetBase(std::vector<Sample>&& samples, std::vector<Label>&& labels)
{
    if (samples.size() == labels.size())
    {
        m_samples = std::move(samples);
        m_labels = std::move(labels);
    }
    else
    {
        throw DifferentSizeException(samples.size(), labels.size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setLabels(gsl::span<const Label> labels)
{
    if (static_cast<size_type>(labels.size()) == m_samples.size())
    {
        m_labels.assign(labels.begin(), labels.end());
    }
    else
    {
        throw DifferentSizeException(labels.size(), m_samples.size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setLabels(std::vector<Label>&& labels)
{
    if (static_cast<size_type>(labels.size()) == m_samples.size())
    {
        m_labels = std::move(labels);
    }
    else
    {
        throw DifferentSizeException(labels.size(), m_samples.size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setSamples(gsl::span<const Sample> samples)
{
    if (static_cast<size_type>(samples.size()) == m_samples.size())
    {
        m_samples.assign(samples.begin(), samples.end());
    }
    else
    {
        throw DifferentSizeException(samples.size(), m_samples.size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::setSamples(std::vector<Sample>&& samples)
{
    if (static_cast<size_type>(samples.size()) == m_samples.size())
    {
        m_samples = std::move(samples);
    }
    else
    {
        throw DifferentSizeException(samples.size(), m_samples.size());
    }
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::addSample(const Sample& sample, const Label& label)
{
    m_samples.emplace_back(sample);
    m_labels.emplace_back(label);
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::addSample(Sample&& sample, Label&& label)
{
    m_samples.emplace_back(std::move(sample));
    m_labels.emplace_back(std::move(label));
}

template <typename Sample, typename Label>
bool DatasetBase<Sample, Label>::empty() const
{
    return size() == 0;
}

template <typename Sample, typename Label>
void DatasetBase<Sample, Label>::reserve(size_t size)
{
    m_samples.reserve(size);
    m_labels.reserve(size);
}

#pragma warning( pop )
}// namespace dataset
