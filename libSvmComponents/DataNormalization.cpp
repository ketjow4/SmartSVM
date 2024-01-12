
#include "DataNormalization.h"
#include "SvmComponentsExceptions.h"
#include "libPlatform/loguru.hpp"

namespace svmComponents
{
void validate(dataset::Dataset<std::vector<float>, float>& trainingData,
              dataset::Dataset<std::vector<float>, float>& validationData,
              dataset::Dataset<std::vector<float>, float>& testData)
{
	if (trainingData.empty())
	{
		throw EmptyDatasetException(DatasetType::Training);
	}
	if (validationData.empty())
	{
		throw EmptyDatasetException(DatasetType::Validation);
	}
	if (testData.empty())
	{
		throw EmptyDatasetException(DatasetType::Test);
	}
}

DataNormalization::DataNormalization(unsigned int numberOfFeatures)
    : m_featureMin(numberOfFeatures)
    , m_featureMax(numberOfFeatures)
    , m_lowerLimit(0.0)
    , m_upperLimit(1.0)
{
}

DataNormalization::DataNormalization(unsigned int numberOfFeatures, float lowerLimit, float upperLimit)

    : m_featureMin(numberOfFeatures)
    , m_featureMax(numberOfFeatures)
    , m_lowerLimit(lowerLimit)
    , m_upperLimit(upperLimit)
{
}

dataset::Dataset<std::vector<float>, float> DataNormalization::scale(dataset::Dataset<std::vector<float>, float>& dataset) const
{
    dataset::Dataset<std::vector<float>, float> scaled;

    for (auto& row : dataset.getSamples())
    {
        std::vector<float> scaledRow;
        scaledRow.reserve(row.size());
        for (auto i = 0u; i < row.size(); i++)
        {
            scaledRow.emplace_back(normalizeValue(m_featureMax[i], m_featureMin[i], row[i]));
        }
        scaled.addSample(std::move(scaledRow), 0);
    }
    scaled.setLabels(dataset.getLabels());

    if(dataset.hasGroups())
    {
        auto groups = dataset.getGroups();
        scaled.setGroups(std::move(groups));
    }

    return scaled;
}


void DataNormalization::normalize(dataset::Dataset<std::vector<float>, float>& trainingData,
                                  dataset::Dataset<std::vector<float>, float>& validationData,
                                  dataset::Dataset<std::vector<float>, float>& testData)
{
    validate(trainingData, validationData, testData);

    const auto& firstSample = trainingData.getSamples()[0];
    m_featureMin = firstSample;
    m_featureMax = firstSample;

	if(m_useDefniedMinMaxValues)
	{
        LOG_F(WARNING, "Using user defined min and max value for data scalling. Designed for 2D datasets usage");
		for(auto i = 0u; i< m_featureMin.size(); ++i)
		{
			m_featureMin[i] = m_min;
			m_featureMax[i] = m_max;
		}
	}
	else
	{
		findScalingFactors(trainingData);
	}
    trainingData = scale(trainingData);
    validationData = scale(validationData);
    testData = scale(testData);
}

const std::vector<float>& DataNormalization::minValuesOfFeatures() const
{
    return m_featureMin;
}

const std::vector<float>& DataNormalization::maxValuesOfFeatures() const
{
    return m_featureMax;
}

void DataNormalization::useDefinedMinMax(float min, float max)
{
    LOG_F(WARNING, "Setting user defined min and max value for data scalling. Designed for 2D datasets usage");
	m_useDefniedMinMaxValues = true;
	m_min = min;
	m_max = max;
	
  /*  std::cout << "Setting user defined min and max value for data scalling. Do you want to continue?\n";
    bool answer = false;
    std::cin >> answer;
	if(answer)
	{
        throw std::exception("break by user");
	}*/
}

float DataNormalization::normalizeValue(float featureMax, float featureMin, float currentValue) const
{
    if (featureMax == featureMin)
    {
        return currentValue;
    }

    if (currentValue == featureMin)
    {
        currentValue = m_lowerLimit;
    }
    else if (currentValue == featureMax)
    {
        currentValue = m_upperLimit;
    }
    else
    {
        currentValue = m_lowerLimit + (m_upperLimit - m_lowerLimit) *
                (currentValue - featureMin) /
                (featureMax - featureMin);
    }

    return currentValue;
}

void DataNormalization::findScalingFactors(const dataset::Dataset<std::vector<float>, float>& trainingData)
{
    // @wdudzik find max and min for each feature
    for (const auto& row : trainingData.getSamples())
    {
        for (auto i = 0u; i < row.size(); i++)
        {
            if (row[i] > m_featureMax[i])
            {
				m_featureMax[i] = row[i];
            }
            else if (row[i] < m_featureMin[i])
            {
				m_featureMin[i] = row[i];
            }
        }
    }
}

StandardScaler::StandardScaler(unsigned /*numberOfFeatures*/)
  /*  : m_means(numberOfFeatures)
	, m_stdDevs(numberOfFeatures)*/
{
}

void StandardScaler::normalize(dataset::Dataset<std::vector<float>, float>& trainingData, dataset::Dataset<std::vector<float>, float>& validationData,
	dataset::Dataset<std::vector<float>, float>& testData)
{
    validate(trainingData, validationData, testData);
  
    findScalingFactors(trainingData);
    
    trainingData = scale(trainingData);
    validationData = scale(validationData);
    testData = scale(testData);
}

const std::vector<float>& StandardScaler::getMeanVector() const
{
    return m_means;
}

const std::vector<float>& StandardScaler::getStdDevV() const
{
    return m_stdDevs;
}

dataset::Dataset<std::vector<float>, float> StandardScaler::scale(dataset::Dataset<std::vector<float>, float>& dataset) const
{
    dataset::Dataset<std::vector<float>, float> scaled;

    for (auto& row : dataset.getSamples())
    {
        std::vector<float> scaledRow;
        scaledRow.reserve(row.size());
        for (auto i = 0u; i < row.size(); i++)
        {
            scaledRow.emplace_back(normalizeValue(m_means[i], m_stdDevs[i], row[i]));
        }
        scaled.addSample(std::move(scaledRow), 0);
    }
    scaled.setLabels(dataset.getLabels());
    return scaled;
}

float StandardScaler::normalizeValue(float mean, float stdDev, float currentValue) const
{
    auto constexpr  epsilon = 1e-10f;
    return (currentValue - mean) / (stdDev + epsilon);
}

std::vector<std::vector<float> > transpose(const gsl::span<const std::vector<float> >& data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    std::vector<std::vector<float>> result(data[0].size(),
                                           std::vector<float>(data.size()));
    for (auto i = 0u; i < data[0].size(); i++)
    {
	    for (auto j = 0u; j < data.size(); j++)
	    {
		    result[i][j] = data[j][i];
	    }
    }
    return result;
}

void StandardScaler::findScalingFactors(const dataset::Dataset<std::vector<float>, float>& trainingData)
{
    auto columns_as_rows = transpose(trainingData.getSamples());
	
    for (const auto& column : columns_as_rows)
    {
        double sum = std::accumulate(std::begin(column), std::end(column), 0.0);
        double m = sum / column.size();

        double accum = 0.0;
        std::for_each(std::begin(column), std::end(column), [&](const double d) {
            accum += (d - m) * (d - m);
            });

        float stdev = static_cast<float>(sqrt(accum / (column.size() - 1)));

        m_means.emplace_back(static_cast<float>(m));
        m_stdDevs.emplace_back(stdev);
    }
}
} // namespace svmComponents
