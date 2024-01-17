#pragma once

#include <string>
#include <vector>
#include "libDataset/Dataset.h"

namespace svmStrategies
{
class NormalizeStrategy
{
public:
	NormalizeStrategy(bool normalize = true)
		: m_normalize(normalize)
	{
	}

    std::string getDescription() const;
    std::tuple<dataset::Dataset<std::vector<float>, float>&,
               dataset::Dataset<std::vector<float>, float>&,
               dataset::Dataset<std::vector<float>, float>&,
               std::vector<float>,
               std::vector<float>> launch(dataset::Dataset<std::vector<float>, float>& trainingData,
                                          dataset::Dataset<std::vector<float>, float>& validationData,
                                          dataset::Dataset<std::vector<float>, float>& testData);
private:
    bool m_normalize;
};
} // namespace svmStrategies
