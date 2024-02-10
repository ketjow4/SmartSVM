#include <libDataset/Dataset.h>
#include "ConfusionMatrix.h"
#include <set>
#include <unordered_set>

#include "BaseSvmChromosome.h"
#include "SvmComponentsExceptions.h"
#include "libPlatform/loguru.hpp"

namespace svmComponents
{




std::array<std::array<uint32_t, 2>, 2> handleGroups(phd::svm::ISvm& svmModel,
                                                    const dataset::Dataset<std::vector<float>, float>& testSamples)
{
	std::array<std::array<uint32_t, 2>, 2> matrix = {0};

	auto answers = svmModel.classifyGroups(testSamples);

	auto labels = testSamples.getLabels();
	auto groups = testSamples.getGroups();

	for (auto [group, answer] : answers)
	{
		if (auto iter = std::find(groups.begin(), groups.end(), group); iter != groups.end())
		{
			auto index = std::distance(groups.begin(), iter);
			matrix[answer][static_cast<uint32_t>(labels[index])]++;
		}
		else
		{
			LOG_F(ERROR, std::string("Group with ID: " + std::to_string(group) + " not found in dataset during confusion matrix calculation").c_str());
			throw std::exception("Group not found in dataset during confusion matrix calculation");
		}
	}

	return matrix;
}



std::array<std::array<uint32_t, 2>, 2> handleGroups(const BaseSvmChromosome& individual,
													const dataset::Dataset<std::vector<float>, float>& testSamples)
{
	auto svmModel = individual.getClassifier();
	return handleGroups(*svmModel, testSamples);
}


ConfusionMatrix::ConfusionMatrix(const BaseSvmChromosome& individual,
                                 const dataset::Dataset<std::vector<float>, float>& testSamples)
	: m_matrix([&]()
	{
		if (testSamples.empty())
		{
			throw EmptyDatasetException(DatasetType::ValidationOrTest);
		}

		auto svmModel = individual.getClassifier();
		if (!svmModel->isTrained())
		{
			throw UntrainedSvmClassifierException();
		}

		if (testSamples.hasGroups())
		{
			return handleGroups(individual, testSamples);
		}

		std::array<std::array<uint32_t, 2>, 2> matrix = {0};

		auto samples = testSamples.getSamples();
		auto targets = testSamples.getLabels();
		int sampleNumber = 0;

		std::for_each(samples.begin(),
		              samples.end(),
		              [&](const gsl::span<const float>& sample)
		              {
							if(svmModel->canClassifyWithOptimalThreshold())
							{
								++matrix[static_cast<int>(svmModel->classifyWithOptimalThreshold(sample))][static_cast<int>(targets[sampleNumber])];
							}
							else
							{
								++matrix[static_cast<int>(svmModel->classify(sample))][static_cast<int>(targets[sampleNumber])];
							}
			              ++sampleNumber;
		              });
		return matrix;
	}())
{
}

ConfusionMatrix::ConfusionMatrix(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& testSamples)
	: m_matrix([&]()
		{
			if (testSamples.empty())
			{
				throw EmptyDatasetException(DatasetType::ValidationOrTest);
			}

			auto& svmModel = svm;
			if (!svmModel.isTrained())
			{
				throw UntrainedSvmClassifierException();
			}

			if (testSamples.hasGroups())
			{
				return handleGroups(svm, testSamples);
			}

			std::array<std::array<uint32_t, 2>, 2> matrix = { 0 };

			auto samples = testSamples.getSamples();
			auto targets = testSamples.getLabels();
			int sampleNumber = 0;

			std::for_each(samples.begin(),
				samples.end(),
				[&](const gsl::span<const float>& sample)
				{
					if (svmModel.canClassifyWithOptimalThreshold())
					{
						++matrix[static_cast<int>(svmModel.classifyWithOptimalThreshold(sample))][static_cast<int>(targets[sampleNumber])];
					}
					else
					{
						//std::cout << svmModel.classify(sample);
						++matrix[static_cast<int>(svmModel.classify(sample))][static_cast<int>(targets[sampleNumber])];
					}
					++sampleNumber;
				});
			return matrix;
		}())
{
}

ConfusionMatrix::ConfusionMatrix(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float>& testSamples, bool /*parrarelCacl*/)
	: m_matrix ( [&]()
{
			if (testSamples.empty())
			{
				throw EmptyDatasetException(DatasetType::ValidationOrTest);
			}

			auto svmModel = individual.getClassifier();
			if (!svmModel->isTrained())
			{
				throw UntrainedSvmClassifierException();
			}

			if (testSamples.hasGroups())
			{
				return handleGroups(individual, testSamples);
			}

			std::array<std::array<uint32_t, 2>, 2> matrix = { 0 };

			auto samples = testSamples.getSamples();
			auto targets = testSamples.getLabels();

			const size_t iterationCount = samples.size();
			auto first = samples.begin();

			#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(iterationCount); i++)
			{
				auto& ithSample = *(first + i);
				//auto ithSample = samples[i];
				//auto result = svmModel->classifyWithOptimalThreshold(ithSample);
				//
				auto result = static_cast<int>(svmModel->classifyWithOptimalThreshold(ithSample));

				//if(static_cast<int>(targets[i]) != result)
				//{
				//	//debug all of the wrong examples for test set
				//	result = static_cast<int>(svmModel->classifyHyperplaneDistance(ithSample));
				//	std::ofstream analyze("D:\\ENSEMBLE_910_scalling_thr_001_k32_RBF\\output_test.txt", std::ios_base::app);
				//	analyze << "Wrong one, true  " << static_cast<int>(targets[i]) << "  predicted " << result << "\n";
				//}
				
				#pragma omp atomic
				++matrix[result][static_cast<int>(targets[i])];
				//++matrix[static_cast<int>(result)][static_cast<int>(targets[i])];
				
			}
			return matrix;
}())
{
	
}
ConfusionMatrix::ConfusionMatrix(uint32_t truePositive, uint32_t trueNegative, uint32_t falsePositive, uint32_t falseNegative)
	: m_matrix([&]()
	{
		std::array<std::array<uint32_t, m_sizeOfBinaryConfusionMatrix>, m_sizeOfBinaryConfusionMatrix> matrix;
		matrix[1][1] = truePositive;
		matrix[0][0] = trueNegative;
		matrix[1][0] = falsePositive;
		matrix[0][1] = falseNegative;
		return matrix;
	}())
{
}

std::ostream& operator<<(std::ostream& out, const ConfusionMatrix& matrix)
{
	return out << matrix.truePositive() << "\t" << matrix.falsePositive() << "\t" << matrix.trueNegative() << "\t" << matrix.falseNegative();
}

std::ostream& operator<<(std::ostream& out, const ConfusionMatrixMulticlass& matrix)
{
	out << matrix.m_numberOfClasses << "\t";
	for(auto& row : matrix.m_matrix)
	{
		for (auto& element : row)
		{
			out << element << "\t";
		}
	}
	return out;
}

ConfusionMatrixMulticlass::ConfusionMatrixMulticlass(const BaseSvmChromosome& individual, const dataset::Dataset<std::vector<float>, float, void>& testSamples)
	: m_numberOfClasses([&]()
	{
		if (testSamples.empty())
		{
			throw EmptyDatasetException(DatasetType::ValidationOrTest);
		}
		auto labels = testSamples.getLabels();
		auto numberOfClasses = std::set<float>(labels.begin(), labels.end()).size();
		return static_cast<unsigned int>(numberOfClasses);
	}())
	, m_matrix([&]()
	{
		if (testSamples.empty())
		{
			throw EmptyDatasetException(DatasetType::ValidationOrTest);
		}
		auto svmModel = individual.getClassifier();
		if (!svmModel->isTrained())
		{
			throw UntrainedSvmClassifierException();
		}
	
		std::vector<std::vector<uint32_t>> matrix;
		matrix.resize(m_numberOfClasses);
		for(auto& row : matrix)
		{
			row.resize(m_numberOfClasses);
		}

		auto samples = testSamples.getSamples();
		auto targets = testSamples.getLabels();
		int sampleNumber = 0;

		std::for_each(samples.begin(),
			samples.end(),
			[&](const gsl::span<const float>& sample)
			{
				++matrix[static_cast<int>(svmModel->classify(sample))][static_cast<int>(targets[sampleNumber])];
				++sampleNumber;
			});
		return matrix;
	}())
{
}

ConfusionMatrixMulticlass::ConfusionMatrixMulticlass(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float, void>& testSamples)
	: m_numberOfClasses([&]()
		{
			if (testSamples.empty())
			{
				throw EmptyDatasetException(DatasetType::ValidationOrTest);
			}
			auto labels = testSamples.getLabels();
			auto numberOfClasses = std::set<float>(labels.begin(), labels.end()).size();
			return static_cast<unsigned int>(numberOfClasses);
		}())
	, m_matrix([&]()
		{
			if (testSamples.empty())
			{
				throw EmptyDatasetException(DatasetType::ValidationOrTest);
			}
			auto& svmModel = svm;
			if (!svmModel.isTrained())
			{
				throw UntrainedSvmClassifierException();
			}

			std::vector<std::vector<uint32_t>> matrix;
			matrix.resize(m_numberOfClasses);
			for (auto& row : matrix)
			{
				row.resize(m_numberOfClasses);
			}

			auto samples = testSamples.getSamples();
			auto targets = testSamples.getLabels();
			int sampleNumber = 0;

			std::for_each(samples.begin(),
				samples.end(),
				[&](const gsl::span<const float>& sample)
				{
					++matrix[static_cast<int>(svmModel.classify(sample))][static_cast<int>(targets[sampleNumber])];
					++sampleNumber;
				});
			return matrix;
		}())
{
}

ConfusionMatrixMulticlass::ConfusionMatrixMulticlass(std::vector<std::vector<uint32_t>> matrix)
	: m_matrix(matrix)
{
}
} // namespace svmComponents
