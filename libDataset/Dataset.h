
#pragma once

#include <vector>
#include <gsl/gsl>
#include "IDataset.h"
#include "DatasetBase.h"

namespace dataset
{
template <typename Sample, typename Label, typename Descritpion = void>
class Dataset : public DatasetBase<Sample, Label>, public IDescription<Descritpion>
{
public:
    Dataset() = default;

    Dataset(Dataset&& dataset) = default;

    Dataset& operator=(Dataset&& dataset) = default;

    Dataset(const Dataset&) = default;

    Dataset& operator=(const Dataset&) = default;

    ~Dataset() = default;

    Dataset(const gsl::span<const Sample> samples, const gsl::span<const Label> labels, const Descritpion& description);

    Dataset(std::vector<Sample>&& samples, std::vector<Label>&& labels, const Descritpion& description);

    const Descritpion& getDatasetDescription() const override;

    void setDatasetDescription(const Descritpion& description) override;

private:
    Descritpion m_datasetDescription;
};

template <typename Sample, typename Label, typename Descritpion>
Dataset<Sample, Label, Descritpion>::Dataset(const gsl::span<const Sample> samples, const gsl::span<const Label> labels, const Descritpion& description)
    : DatasetBase(samples, labels)
{
    m_datasetDescription = description;
}

template <typename Sample, typename Label, typename Descritpion>
Dataset<Sample, Label, Descritpion>::Dataset(std::vector<Sample>&& samples, std::vector<Label>&& labels, const Descritpion& description)
    : DatasetBase(samples, labels)
{
    m_datasetDescription = description;
}

template <typename Sample, typename Label, typename Descritpion>
const Descritpion& Dataset<Sample, Label, Descritpion>::getDatasetDescription() const
{
    return m_datasetDescription;
}

template <typename Sample, typename Label, typename Descritpion>
void Dataset<Sample, Label, Descritpion>::setDatasetDescription(const Descritpion& description)
{
    m_datasetDescription = description;
}

template <typename Sample, typename Label>
class Dataset<Sample, Label, void> : public DatasetBase<Sample, Label>
{
public:
    Dataset() = default;

    Dataset(Dataset&& dataset) = default;

    Dataset& operator=(Dataset&& dataset) = default;

    Dataset(const Dataset&) = default;

    Dataset& operator=(const Dataset&) = default;

    ~Dataset() = default;

    Dataset(const gsl::span<const Sample> samples, const gsl::span<const Label> labels);

    Dataset(std::vector<Sample>&& samples, std::vector<Label>&& labels);

    Dataset(std::vector<Sample>&& samples, std::vector<Label>&& labels, std::vector<float>&& groups);

    bool hasGroups() const;

    const std::vector<float>& getGroups() const;

    float getGroups(int index) const;

    void setGroups(std::vector<float>&& groups);



private:
    bool m_hasGroups = false;
    std::vector<float> m_groups;

};

template <typename Sample, typename Label>
Dataset<Sample, Label, void>::Dataset(const gsl::span<const Sample> samples, const gsl::span<const Label> labels)
    : DatasetBase(samples, labels)
	, m_hasGroups(false)
{
    
}

template <typename Sample, typename Label>
Dataset<Sample, Label, void>::Dataset(std::vector<Sample>&& samples, std::vector<Label>&& labels)
    : DatasetBase(samples, labels)
    , m_hasGroups(false)
{
}

template <typename Sample, typename Label>
Dataset<Sample, Label, void>::Dataset(std::vector<Sample>&& samples, std::vector<Label>&& labels, std::vector<float>&& groups)
    : DatasetBase(samples, labels)
    , m_hasGroups(true)
	, m_groups(groups)
{
}

template <typename Sample, typename Label>
bool Dataset<Sample, Label, void>::hasGroups() const
{
    return m_hasGroups;
}

template <typename Sample, typename Label>
const std::vector<float>& Dataset<Sample, Label, void>::getGroups() const
{
    return m_groups;
}

template <typename Sample, typename Label>
float Dataset<Sample, Label, void>::getGroups(int index) const
{
    if (index < m_groups.size())
    {
        return m_groups[index];
    }
    throw OutOfBoundsException(index, m_groups.size());
	
}

template <typename Sample, typename Label>
void Dataset<Sample, Label, void>::setGroups(std::vector<float>&& groups)
{
    m_groups = std::move(groups);
    m_hasGroups = true;
}
}// namespace dataset
