
#pragma once

#include <vector>
#include "libDataset/Dataset.h"

namespace svmComponents
{
void validate(dataset::Dataset<std::vector<float>, float>& trainingData,
              dataset::Dataset<std::vector<float>, float>& validationData,
              dataset::Dataset<std::vector<float>, float>& testData);


class DataNormalization
{
public:
    explicit DataNormalization(unsigned int numberOfFeatures);
    DataNormalization(unsigned int numberOfFeatures, float lowerLimit, float upperLimit);

    void normalize(dataset::Dataset<std::vector<float>, float>& trainingData,
                   dataset::Dataset<std::vector<float>, float>& validationData,
                   dataset::Dataset<std::vector<float>, float>& testData);

    const std::vector<float>& minValuesOfFeatures() const;
    const std::vector<float>& maxValuesOfFeatures() const;

	//use only for 2D dataset
	static void useDefinedMinMax(float min, float max);

private:
    dataset::Dataset<std::vector<float>, float> scale(dataset::Dataset<std::vector<float>, float>& dataset) const;
   

    float normalizeValue(float featureMax, float featureMin, float currentValue) const;
    void findScalingFactors(const dataset::Dataset<std::vector<float>, float>& trainingData);

    std::vector<float> m_featureMin;
    std::vector<float> m_featureMax;

    const float m_lowerLimit;
    const float m_upperLimit;

    inline static bool m_useDefniedMinMaxValues = false;
	inline static float m_min;
	inline static float m_max;
};


class StandardScaler
{
public:
    explicit StandardScaler(unsigned int numberOfFeatures);
   
    void normalize(dataset::Dataset<std::vector<float>, float>& trainingData,
        dataset::Dataset<std::vector<float>, float>& validationData,
        dataset::Dataset<std::vector<float>, float>& testData);

    const std::vector<float>& getMeanVector() const;
    const std::vector<float>& getStdDevV() const;

	
private:
    dataset::Dataset<std::vector<float>, float> scale(dataset::Dataset<std::vector<float>, float>& dataset) const;

    float normalizeValue(float mean, float stdDev, float currentValue) const;
    void findScalingFactors(const dataset::Dataset<std::vector<float>, float>& trainingData);

    std::vector<float> m_means;
    std::vector<float> m_stdDevs;
};
} // namespace svmComponents
