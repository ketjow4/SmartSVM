
#pragma once

#include "IReadOnlyDataset.h"
#include "DatasetViewBase.h"
#include "Dataset.h"

namespace dataset
{
template <typename Sample, typename Label, typename Description = void>
class DatasetView : public DatasetViewBase<Sample, Label>, public IReadOnlyDescritpion<Description>
{
public:
    using typename IReadOnlyDataset<Sample, Label>::size_type;

    DatasetView(const Dataset<Sample, Label, Description>& dataset);

    DatasetView() = default;

    DatasetView(DatasetView&& dataset) = default;

    DatasetView& operator=(DatasetView&& dataset) = default;

    DatasetView(const DatasetView&) = default;

    DatasetView& operator=(const DatasetView&) = default;

    const Description& getDatasetDescription() const override;
private:
    std::shared_ptr<Description> m_datasetDescription;
};

template <typename Sample, typename Label, typename Description>
DatasetView<Sample, Label, Description>::DatasetView(const Dataset<Sample, Label, Description>& dataset)
    : DatasetViewBase(dataset)
    , m_datasetDescription(std::make_shared<Description>(dataset.getDatasetDescription()))
{
}

template <typename Sample, typename Label, typename Description>
const Description& DatasetView<Sample, Label, Description>::getDatasetDescription() const
{
    if (m_datasetDescription)
    {
        return *m_datasetDescription.get();
    }
    throw DescriptionIsNullInDatasetView();
}

template <typename Sample, typename Label>
class DatasetView<Sample, Label, void> : public DatasetViewBase<Sample, Label>
{
public:
    using typename IReadOnlyDataset<Sample, Label>::size_type;

    DatasetView() = default;

    DatasetView(DatasetView&& dataset) = default;

    DatasetView& operator=(DatasetView&& dataset) = default;

    DatasetView(const DatasetView&) = default;

    DatasetView& operator=(const DatasetView&) = default;

    ~DatasetView() = default;

    DatasetView(const Dataset<Sample, Label>& dataset);
};

template <typename Sample, typename Label>
DatasetView<Sample, Label, void>::DatasetView(const Dataset<Sample, Label>& dataset)
    : DatasetViewBase(dataset)
{
}
} //namespace dataset
