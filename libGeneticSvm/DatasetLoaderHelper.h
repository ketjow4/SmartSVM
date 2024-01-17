#pragma once

#include "IDatasetLoader.h"

namespace genetic
{
    class DatasetLoaderHelper : public IDatasetLoader
    {
    public:
	    DatasetLoaderHelper(const dataset::Dataset<std::vector<float>, float>& tr,
	                        const dataset::Dataset<std::vector<float>, float>& val,
	                        const dataset::Dataset<std::vector<float>, float>& test);

	    const dataset::Dataset<std::vector<float>, float>& getTraningSet() override;
	    const dataset::Dataset<std::vector<float>, float>& getValidationSet() override;
	    const dataset::Dataset<std::vector<float>, float>& getTestSet() override;

	    bool isDataLoaded() const override;
	    const std::vector<float>& scalingVectorMin() override;
	    const std::vector<float>& scalingVectorMax() override;

    private:
        std::vector<float> m_stub;

        const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
        const dataset::Dataset<std::vector<float>, float>& m_validationSet;
        const dataset::Dataset<std::vector<float>, float>& m_testSet;
    };
}
