#pragma once

#include "SvmCustomKernelChromosome.h"
#include "ISvmTraining.h"
#include "SvmComponentsExceptions.h"
#include "SvmLib/SvmFactory.h"
#include "SvmUtils.h"
#include "SvmConfigStructures.h"
#include "libPlatform/EnumStringConversions.h"
#include "SvmLib/libSvmImplementation.h"

namespace svmComponents
{
	class SvmTrainingCustomKernel : public ISvmTraining<SvmCustomKernelChromosome>
	{
	public:
		explicit SvmTrainingCustomKernel(SvmAlgorithmConfiguration svmConfig,
			bool probabilityOutputNeeded, std::string kernelName, bool trainAlpha);

		void trainPopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population,
			const dataset::Dataset<std::vector<float>, float>& trainingData,
			const std::vector<Gene>& frozenOnes);

		void trainPopulation(geneticComponents::Population<SvmCustomKernelChromosome>& population,
			const dataset::Dataset<std::vector<float>, float>& trainingData) override
		{
			if (population.empty())
			{
				throw geneticComponents::PopulationIsEmptyException();
			}
			if (trainingData.empty())
			{
				throw EmptyDatasetException(DatasetType::Training);
			}

			const size_t iterationCount = std::distance(population.begin(), population.end());
			auto first = population.begin();

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(iterationCount); i++)
			{
				try
				{
					auto& individual = *(first + i);
					//auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);
					auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType);

					auto cl = reinterpret_cast<phd::svm::libSvmImplementation*>(classifier.get());
					cl->setAlphaTraining(m_trainAlpha);

					svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
					classifier->setKernel(m_kernelType);
					classifier->setGammas(individual.getGammas());
					classifier->setC(individual.getC());

					auto individualTraningSet = individual.convertChromosome(trainingData);
					classifier->train(individualTraningSet, m_probabilityOutputNeeded);

				
					if (individual.hasFeatures())
					{
						classifier->setFeatureSet(individual.getFeatures(), static_cast<int>(trainingData.getSample(0).size()));
					}
					else
					{
						std::vector<Feature> vec;
						vec.resize(static_cast<int>(trainingData.getSample(0).size()));
						std::uint64_t j = 0;
						std::iota(vec.begin(), vec.end(), j);
						classifier->setFeatureSet(vec, static_cast<int>(trainingData.getSample(0).size()));
					}
					individual.updateClassifier(std::move(classifier));

				}
				catch (const UnsupportedKernelTypeException & exception)
				{
#pragma omp critical
					m_lastException = std::make_exception_ptr(exception);
				}
			}
			if (m_lastException)
			{
				std::rethrow_exception(m_lastException);
			}
		}

	private:
		const SvmAlgorithmConfiguration m_svmConfig;
		const bool m_probabilityOutputNeeded;
		std::exception_ptr m_lastException;
		std::unordered_map<std::string, phd::svm::KernelTypes> m_translationsGeneration;
		phd::svm::KernelTypes m_kernelType;
		bool m_trainAlpha;
	};

	inline SvmTrainingCustomKernel::SvmTrainingCustomKernel(SvmAlgorithmConfiguration svmConfig, bool /*probabilityOutputNeeded*/, std::string kernelName, bool trainAlpha)
		: m_svmConfig(svmConfig)
		, m_probabilityOutputNeeded(false)
		, m_trainAlpha(trainAlpha)
	{
		m_translationsGeneration =
		{
			{"RBF_CUSTOM", phd::svm::KernelTypes::Rbf_custom},
			{"RBF_SUM", phd::svm::KernelTypes::RBF_SUM},
			{"RBF_SUM_DIV2", phd::svm::KernelTypes::RBF_SUM_DIV2},
			{"RBF_DIV", phd::svm::KernelTypes::RBF_DIV},
			{"RBF_MAX", phd::svm::KernelTypes::RBF_MAX},
			{"RBF_MIN", phd::svm::KernelTypes::RBF_MIN},
			{"RBF_SUM_2_KERNELS", phd::svm::KernelTypes::RBF_SUM_2_KERNELS},
			{"RBF_LINEAR", phd::svm::KernelTypes::RBF_LINEAR},
			{"RBF_LINEAR_SINGLE", phd::svm::KernelTypes::RBF_LINEAR_SINGLE},
			{"RBF_LINEAR_MAX", phd::svm::KernelTypes::RBF_LINEAR_MAX},
			{"RBF_LINEAR_MIN", phd::svm::KernelTypes::RBF_LINEAR_MIN},
			{"RBF_LINEAR_SUM_2_KERNELS", phd::svm::KernelTypes::RBF_LINEAR_SUM_2_KERNELS},
			
	};

	m_kernelType = platform::stringToEnum(kernelName, m_translationsGeneration);
}

inline void SvmTrainingCustomKernel::trainPopulation(Population<SvmCustomKernelChromosome>& population,
	const dataset::Dataset<std::vector<float>, float>& trainingData,
	const std::vector<Gene>& frozenOnes)
{
	if (population.empty())
	{
		throw geneticComponents::PopulationIsEmptyException();
	}
	if (trainingData.empty())
	{
		throw EmptyDatasetException(DatasetType::Training);
	}

	const size_t iterationCount = std::distance(population.begin(), population.end());
	auto first = population.begin();

#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(iterationCount); i++)
	{
		try
		{
			auto& individual = *(first + i);
			//auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);
			auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType);

			auto cl = reinterpret_cast<phd::svm::libSvmImplementation*>(classifier.get());
			cl->setAlphaTraining(m_trainAlpha);
			
			std::vector<Gene> newOne;
			auto current = individual.getDataset();
			newOne.reserve(current.size() + frozenOnes.size()); // preallocate memory
			newOne.insert(newOne.end(), current.begin(), current.end());
			newOne.insert(newOne.end(), frozenOnes.begin(), frozenOnes.end());
			auto debugCopy = newOne;


			SvmCustomKernelChromosome withFrozenOnes;
			if (individual.hasFeatures())
			{
				withFrozenOnes = SvmCustomKernelChromosome{ std::move(newOne), individual.getC(), individual.getFeatures() };
			}
			else
			{
				withFrozenOnes = SvmCustomKernelChromosome{ std::move(newOne), individual.getC() };
			}
			

			svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
			classifier->setKernel(m_kernelType);
			classifier->setGammas(withFrozenOnes.getGammas());
			classifier->setC(withFrozenOnes.getC());

			auto individualTraningSet = withFrozenOnes.convertChromosome(trainingData);
			classifier->train(individualTraningSet, m_probabilityOutputNeeded);

			if(individual.hasFeatures())
			{
				classifier->setFeatureSet(individual.getFeatures(), static_cast<int>(trainingData.getSample(0).size()));
			}
			else
			{
				std::vector<Feature> vec;
				vec.resize(static_cast<int>(trainingData.getSample(0).size()));
				std::uint64_t j = 0;
				std::iota(vec.begin(), vec.end(), j);
				classifier->setFeatureSet(vec, static_cast<int>(trainingData.getSample(0).size()));
			}
		
			
			//classifier->setFeatureSet(individual.getFeatures(), static_cast<int>(trainingData.getSample(0).size()));

			/*auto res2 = reinterpret_cast<phd::svm::SvmLibImplementation*>(classifier.get());
			if (res2->isMaxIterReached())
			{
#pragma omp critical
				std::cout << "Max iter, investigate what is happening:\n";
	
				for (auto vec : debugCopy)
				{
					std::cout << vec.id << "\t" << vec.gamma <<  "\t" << trainingData.getSamples()[vec.id][0] << "\t" << trainingData.getSamples()[vec.id][1] << "\n";
				}

				auto cls2 = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType);

				SvmCustomKernelChromosome withFrozenOnes2{ std::move(debugCopy), individual.getC() };

				svmUtils::setupSvmTerminationCriteria(*cls2, m_svmConfig);
				cls2->setKernel(phd::svm::KernelTypes::Rbf_custom);
				cls2->setGammas(withFrozenOnes2.getGammas());
				cls2->setC(withFrozenOnes2.getC());
				cls2->setTerminationCriteria(cv::TermCriteria(0, 0, 1e-1));

				auto individualTraningSet2 = withFrozenOnes2.convertChromosome(trainingData);
				cls2->train(individualTraningSet2, m_probabilityOutputNeeded);

				auto res3 = reinterpret_cast<phd::svm::SvmLibImplementation*>(classifier.get());
				if (res3->isMaxIterReached())
				{
					std::cout << "Epsilon didn't help!!!\n\n\n";
				}
				else
					std::cout << "Epsilon helped!!!\n\n\n";

				std::cout << "End of problematic svm\n\n\n";
			}*/

			individual.updateClassifier(std::move(classifier));
		}
		catch (const UnsupportedKernelTypeException& exception)
		{
#pragma omp critical
			m_lastException = std::make_exception_ptr(exception);
		}
	}
	if (m_lastException)
	{
		std::rethrow_exception(m_lastException);
	}
}





















#include "libSvmComponents/SvmCustomKernelFeaturesSelectionChromosome.h"

class SvmTrainingCustomKernelFS : public ISvmTraining<SvmCustomKernelFeaturesSelectionChromosome>
{
public:
	explicit SvmTrainingCustomKernelFS(SvmAlgorithmConfiguration svmConfig,
		bool probabilityOutputNeeded, std::string kernelName, bool trainAlpha);

	void trainPopulation(geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>& population,
		const dataset::Dataset<std::vector<float>, float>& trainingData,
		const std::vector<Gene>& frozenOnes);

	void trainPopulation(geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>& population,
		const dataset::Dataset<std::vector<float>, float>& trainingData) override
	{
		if (population.empty())
		{
			throw geneticComponents::PopulationIsEmptyException();
		}
		if (trainingData.empty())
		{
			throw EmptyDatasetException(DatasetType::Training);
		}

		const size_t iterationCount = std::distance(population.begin(), population.end());
		auto first = population.begin();

#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(iterationCount); i++)
		{
			try
			{
				auto& individual = *(first + i);
				//auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);
				auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType);

				auto cl = reinterpret_cast<phd::svm::libSvmImplementation*>(classifier.get());
				cl->setAlphaTraining(m_trainAlpha);

				svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
				classifier->setKernel(m_kernelType);
				classifier->setGammas(individual.getKernel().getGammas());
				classifier->setC(individual.getKernel().getC());

				auto individualTraningSet = individual.convertChromosome(trainingData);
				classifier->train(individualTraningSet, m_probabilityOutputNeeded);
				classifier->setFeatureSet(individual.getFeatures(), static_cast<int>(trainingData.getSample(0).size()));
				individual.updateClassifier(std::move(classifier));

			}
			catch (const UnsupportedKernelTypeException& exception)
			{
#pragma omp critical
				m_lastException = std::make_exception_ptr(exception);
			}
		}
		if (m_lastException)
		{
			std::rethrow_exception(m_lastException);
		}
	}

private:
	const SvmAlgorithmConfiguration m_svmConfig;
	const bool m_probabilityOutputNeeded;
	std::exception_ptr m_lastException;
	std::unordered_map<std::string, phd::svm::KernelTypes> m_translationsGeneration;
	phd::svm::KernelTypes m_kernelType;
	bool m_trainAlpha;
};

inline SvmTrainingCustomKernelFS::SvmTrainingCustomKernelFS(SvmAlgorithmConfiguration svmConfig, bool /*probabilityOutputNeeded*/, std::string kernelName, bool trainAlpha)
	: m_svmConfig(svmConfig)
	, m_probabilityOutputNeeded(false)
	, m_trainAlpha(trainAlpha)
{
	m_translationsGeneration =
	{
		{"RBF_CUSTOM", phd::svm::KernelTypes::Rbf_custom},
		{"RBF_SUM", phd::svm::KernelTypes::RBF_SUM},
		{"RBF_SUM_DIV2", phd::svm::KernelTypes::RBF_SUM_DIV2},
		{"RBF_DIV", phd::svm::KernelTypes::RBF_DIV},
		{"RBF_MAX", phd::svm::KernelTypes::RBF_MAX},
		{"RBF_MIN", phd::svm::KernelTypes::RBF_MIN},
		{"RBF_SUM_2_KERNELS", phd::svm::KernelTypes::RBF_SUM_2_KERNELS},
		{"RBF_LINEAR", phd::svm::KernelTypes::RBF_LINEAR},
		{"RBF_LINEAR_SINGLE", phd::svm::KernelTypes::RBF_LINEAR_SINGLE},
		{"RBF_LINEAR_MAX", phd::svm::KernelTypes::RBF_LINEAR_MAX},
		{"RBF_LINEAR_MIN", phd::svm::KernelTypes::RBF_LINEAR_MIN},
		{"RBF_LINEAR_SUM_2_KERNELS", phd::svm::KernelTypes::RBF_LINEAR_SUM_2_KERNELS},
	};

	m_kernelType = platform::stringToEnum(kernelName, m_translationsGeneration);
}

inline void SvmTrainingCustomKernelFS::trainPopulation(Population<SvmCustomKernelFeaturesSelectionChromosome>& population,
	const dataset::Dataset<std::vector<float>, float>& trainingData,
	const std::vector<Gene>& frozenOnes)
{
	if (population.empty())
	{
		throw geneticComponents::PopulationIsEmptyException();
	}
	if (trainingData.empty())
	{
		throw EmptyDatasetException(DatasetType::Training);
	}

	const size_t iterationCount = std::distance(population.begin(), population.end());
	auto first = population.begin();

#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(iterationCount); i++)
	{
		try
		{
			auto& individual = *(first + i);
			//auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType, m_svmConfig.m_groupPropagationMethod);
			auto classifier = phd::svm::SvmFactory::create(m_svmConfig.m_implementationType);

			auto cl = reinterpret_cast<phd::svm::libSvmImplementation*>(classifier.get());
			cl->setAlphaTraining(m_trainAlpha);

			std::vector<Gene> newOne;
			auto current = individual.getKernel().getDataset();
			newOne.reserve(current.size() + frozenOnes.size()); // preallocate memory
			newOne.insert(newOne.end(), current.begin(), current.end());
			newOne.insert(newOne.end(), frozenOnes.begin(), frozenOnes.end());
			auto debugCopy = newOne;
			SvmCustomKernelChromosome withFrozenOnes{ std::move(newOne), individual.getKernel().getC() };

			svmUtils::setupSvmTerminationCriteria(*classifier, m_svmConfig);
			classifier->setKernel(m_kernelType);
			classifier->setGammas(withFrozenOnes.getGammas());
			classifier->setC(withFrozenOnes.getC());

			auto individualTraningSet = withFrozenOnes.convertChromosome(trainingData);
			classifier->train(individualTraningSet, m_probabilityOutputNeeded);
			classifier->setFeatureSet(individual.getFeatures(), static_cast<int>(trainingData.getSample(0).size()));

			individual.updateClassifier(std::move(classifier));
		}
		catch (const UnsupportedKernelTypeException& exception)
		{
#pragma omp critical
			m_lastException = std::make_exception_ptr(exception);
		}
	}
	if (m_lastException)
	{
		std::rethrow_exception(m_lastException);
	}
}

} // namespace svmComponents
