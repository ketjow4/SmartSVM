#include "DatasetLoaderHelper.h"

namespace genetic
{
DatasetLoaderHelper::DatasetLoaderHelper(const dataset::Dataset<std::vector<float>, float>& tr,
                                         const dataset::Dataset<std::vector<float>, float>& val,
                                         const dataset::Dataset<std::vector<float>, float>& test)
	: m_trainingSet(tr)
	, m_validationSet(val)
	, m_testSet(test)
{
}

const dataset::Dataset<std::vector<float>, float>& DatasetLoaderHelper::getTraningSet()
{
	return m_trainingSet;
}

const dataset::Dataset<std::vector<float>, float>& DatasetLoaderHelper::getValidationSet()
{
	return m_validationSet;
}

const dataset::Dataset<std::vector<float>, float>& DatasetLoaderHelper::getTestSet()
{
	return m_testSet;
}

bool DatasetLoaderHelper::isDataLoaded() const
{
	return true;
}

const std::vector<float>& DatasetLoaderHelper::scalingVectorMin()
{
	return m_stub;
}

const std::vector<float>& DatasetLoaderHelper::scalingVectorMax()
{
	return m_stub;
}
}
