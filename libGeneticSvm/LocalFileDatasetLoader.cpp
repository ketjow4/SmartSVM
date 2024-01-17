#include <mutex>
#include "libSvmStrategies/NormalizeStrategy.h"
//#include "libStrategies/TabularDataProviderStrategy.h"
#include "libDataset/TabularDataProvider.h"
#include "LocalFileDatasetLoader.h"
#include "WorkflowUtils.h"
#include "libPlatform/loguru.hpp"
#include <future>

//validation set dynamic selection in runtime
#include "libRandom/MersenneTwister64Rng.h"
#include "libSvmComponents/SvmUtils.h"

namespace genetic
{
	dataset::Dataset<std::vector<float>, float> joinSets3(const dataset::Dataset<std::vector<float>, float>& tr,
		const dataset::Dataset<std::vector<float>, float>& val,
		const dataset::Dataset<std::vector<float>, float>& test)
	{
		dataset::Dataset<std::vector<float>, float> joinedAll;

		for (auto i = 0u; i < tr.size(); ++i)
		{
			joinedAll.addSample(tr.getSample(i), tr.getLabel(i));
		}

		for (auto i = 0u; i < val.size(); ++i)
		{
			joinedAll.addSample(val.getSample(i), val.getLabel(i));
		}
		for (auto i = 0u; i < test.size(); ++i)
		{
			joinedAll.addSample(test.getSample(i), test.getLabel(i));
		}

		return joinedAll;
	}

LocalFileDatasetLoader::LocalFileDatasetLoader(const std::filesystem::path& trainingSetPath,
                                               const std::filesystem::path& validationSetPath,
                                               const std::filesystem::path& testSetPath,
                                               bool normalize,
                                               bool resampleTrainingValidation)
	: m_isDataLoaded(false)
	, m_trainingSetPath(trainingSetPath)
	, m_validationSetPath(validationSetPath)
	, m_testSetPath(testSetPath)
	, m_normalize(normalize)
	, m_resampleTrainingValidation(resampleTrainingValidation)

{
}

//assumes the name of files, used only for debugging models
LocalFileDatasetLoader::LocalFileDatasetLoader(const std::filesystem::path& datasetPath)
	: m_isDataLoaded(false)
	, m_trainingSetPath(datasetPath / "train.csv")
	, m_validationSetPath(datasetPath / "validation.csv")
	, m_testSetPath(datasetPath / "test.csv")
{
	m_resampleTrainingValidation = true;
	m_normalize = true;
}

LocalFileDatasetLoader::LocalFileDatasetLoader(const std::filesystem::path& trainingSetPath,
                                               const std::filesystem::path& validationSetPath,
                                               const std::filesystem::path& testSetPath,
                                               std::vector<bool> featureMask)
	: LocalFileDatasetLoader(trainingSetPath, validationSetPath, testSetPath)
{
	m_resampleTrainingValidation = true;
	m_normalize = true;
	m_featureMask = std::move(featureMask);
}

void LocalFileDatasetLoader::loadDataAndNormalize()
{
	try
	{
		/*std::future< dataset::Dataset<std::vector<float>, float>> ret = std::async(&strategies::TabularDataProviderStrategy::launch, &m_loadTraningElement, m_trainingSetPath);
		std::future< dataset::Dataset<std::vector<float>, float>> ret2 = std::async(&strategies::TabularDataProviderStrategy::launch, &m_loadValidationElement, m_validationSetPath);
		std::future< dataset::Dataset<std::vector<float>, float>> ret3 = std::async(&strategies::TabularDataProviderStrategy::launch, &m_loadTestElement, m_testSetPath);
		

		m_traningSet = ret.get();
		m_validationSet = ret2.get();
		m_testSet = ret3.get();*/
		phd::data::TabularDataProvider dataProvider;


		m_traningSet = dataProvider.loadData(m_trainingSetPath);
		m_validationSet = dataProvider.loadData(m_validationSetPath);
		m_testSet = dataProvider.loadData(m_testSetPath);


		//TODO change what is returned as dataset type is changed! groups should be set during reading the data inside tabular data provider
		/*auto [trDataset, trGroup] = m_loadTraningElement.launch(m_trainingSetPath);
		m_traningSet = trDataset;

		auto [validationDataset, valGroup] = m_loadValidationElement.launch(m_validationSetPath);
		m_validationSet = validationDataset;

		auto [testDataset, testGroup] = m_loadTestElement.launch(m_testSetPath);
		m_testSet = testDataset;

		m_traningSet.setGroups(std::move(trGroup));
		m_validationSet.setGroups(std::move(valGroup));
		m_testSet.setGroups(std::move(testGroup));*/

		//m_testSet = joinSets3(m_traningSet, m_validationSet, m_testSet);
		
		if (m_normalize)
		{
			auto [traningSet, validationSet, testSet, minValuesOfFeatures, maxValuesOfFeatures] =
					m_normalizeElement.launch(m_traningSet, m_validationSet, m_testSet);

			m_traningSet = traningSet;
			m_validationSet = validationSet;
			m_testSet = testSet;
			m_scallingMin = minValuesOfFeatures;
			m_scallingMax = maxValuesOfFeatures;
		}

		//@wdudzik apply feature mask to datasets
		if (!m_featureMask.empty())
		{
			svmComponents::SvmFeatureSetChromosome chromosome(std::move(m_featureMask));

			m_traningSet = chromosome.convertChromosome(m_traningSet);
			m_validationSet = chromosome.convertChromosome(m_validationSet);
			m_testSet = chromosome.convertChromosome(m_testSet);
		}

		//unified training with validation
		if (m_resampleTrainingValidation && !m_traningSet.hasGroups())
		{
			dataset::Dataset<std::vector<float>, float> joined_tr_val;

			if (m_traningSet.size() != m_validationSet.size())
			{
				for (auto i = 0u; i < m_traningSet.size(); ++i)
				{
					joined_tr_val.addSample(m_traningSet.getSample(i), m_traningSet.getLabel(i));
				}

				for (auto i = 0u; i < m_validationSet.size(); ++i)
				{
					joined_tr_val.addSample(m_validationSet.getSample(i), m_validationSet.getLabel(i));
				}

				/*if(m_validationSet.hasGroups() && m_traningSet.hasGroups())
				{
					auto t = m_traningSet.getGroups();
					auto v = m_validationSet.getGroups();

					t.insert(t.end(), v.begin(), v.end());
					joined_tr_val.setGroups(std::move(t));
				}*/

				

			}
			else
			{
				joined_tr_val = m_traningSet;
			}

			//  m_traningSet = joined_tr_val;
			//  m_validationSet = joined_tr_val;

			auto classCount = svmComponents::svmUtils::countLabels(2, joined_tr_val);

			if (std::all_of(classCount.begin(), classCount.end(), [](auto value)
			{
				return value > 0;
			}))
			{
				auto validationSizeN = std::round(classCount[0] * 0.25);
				auto validationSizeP = std::round(classCount[1] * 0.25);

				//std::vector<uint64_t> indexsValidation;
				std::unordered_set<uint64_t> indexsSet;
				dataset::Dataset<std::vector<float>, float> newValidation;
				//std::vector<float> trainingGroups;
				//std::vector<float> validationGroups;

				auto rngEngine = std::make_unique<random::MersenneTwister64Rng>(0);
				auto randomID = std::uniform_int_distribution<int>(0, static_cast<int>(joined_tr_val.size() - 1));

				int numberOfExamples = 0;
				while (numberOfExamples < validationSizeN)
				{
					auto index = rngEngine->getRandom(randomID);
					if (joined_tr_val.getLabel(index) == 0
						&& indexsSet.emplace(static_cast<int>(index)).second)
					{
						newValidation.addSample(joined_tr_val.getSample(index), joined_tr_val.getLabel(index));
						//validationGroups.emplace_back(joined_tr_val.getGroups(index));
						numberOfExamples++;
					}
				}

				numberOfExamples = 0;
				while (numberOfExamples < validationSizeP)
				{
					auto index = rngEngine->getRandom(randomID);
					if (joined_tr_val.getLabel(index) == 1
						&& indexsSet.emplace(static_cast<int>(index)).second)
					{
						newValidation.addSample(joined_tr_val.getSample(index), joined_tr_val.getLabel(index));
						//validationGroups.emplace_back(joined_tr_val.getGroups(index));
						numberOfExamples++;
					}
				}

				dataset::Dataset<std::vector<float>, float> newTraining;

				for (auto i = 0u; i < joined_tr_val.size(); ++i)
				{
					if (indexsSet.emplace(static_cast<int>(i)).second)
					{
						newTraining.addSample(joined_tr_val.getSample(i), joined_tr_val.getLabel(i));
						//trainingGroups.emplace_back(joined_tr_val.getGroups(i));
					}
				}

				//newTraining.setGroups(std::move(trainingGroups));
				//newValidation.setGroups(std::move(validationGroups));
				m_traningSet = newTraining;
				m_validationSet = newValidation;

				auto newTR = svmComponents::svmUtils::countLabels(2, m_traningSet);
				auto newVal = svmComponents::svmUtils::countLabels(2, m_validationSet);
			}
		}

		m_isDataLoaded = true;
		LOG_F(INFO, "Successful loaded and normalized datasets");
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << exception.what();
	}
}

const dataset::Dataset<std::vector<float>, float>& LocalFileDatasetLoader::getTraningSet()
{
	std::call_once(m_shouldLoadData, [&]()
	{
		loadDataAndNormalize();
	});
	return m_traningSet;
}

const dataset::Dataset<std::vector<float>, float>& LocalFileDatasetLoader::getValidationSet()
{
	std::call_once(m_shouldLoadData, [&]()
	{
		loadDataAndNormalize();
	});

	return m_validationSet;
}

const dataset::Dataset<std::vector<float>, float>& LocalFileDatasetLoader::getTestSet()
{
	std::call_once(m_shouldLoadData, [&]()
	{
		loadDataAndNormalize();
	});
	return m_testSet;
}

bool LocalFileDatasetLoader::isDataLoaded() const
{
	return m_isDataLoaded;
}

const std::vector<float>& LocalFileDatasetLoader::scalingVectorMin()
{
	std::call_once(m_shouldLoadData, [&]()
	{
		loadDataAndNormalize();
	});
	return m_scallingMin;
}

const std::vector<float>& LocalFileDatasetLoader::scalingVectorMax()
{
	std::call_once(m_shouldLoadData, [&]()
	{
		loadDataAndNormalize();
	});
	return m_scallingMax;
}
} // namespace genetic
