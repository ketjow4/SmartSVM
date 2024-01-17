# include "EnsembleUtils.h"

namespace genetic
{
std::pair<std::vector<svmComponents::DatasetVector>, std::vector<uint64_t>> getCertainDataset(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                                              const std::vector<uint64_t>& ids,
                                                                                              std::shared_ptr<phd::svm::ISvm> svm)
{
	std::vector<svmComponents::DatasetVector> certainTrainingSet;
	std::vector<uint64_t> certainIds;

	auto samples = trainingSet.getSamples();
	auto labels = trainingSet.getLabels();

	for (auto i = 0u; i < samples.size(); ++i)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());
		auto result = res->classifyWithCertainty(samples[i]);
		if (result != -100) //if we are sure (so either 0 or 1)
		{
			certainIds.emplace_back(ids[i]);
			certainTrainingSet.emplace_back(svmComponents::DatasetVector{ids[i], labels[i]});
		}
	}

	return {certainTrainingSet, certainIds};
}

std::pair<std::vector<DatasetVector>, std::vector<uint64_t>> getSVsIds(const dataset::Dataset<std::vector<float>, float>& trainingDataset,
                                                                       const std::vector<uint64_t>& ids, std::shared_ptr<phd::svm::ISvm> svm)
{
	const auto samples = trainingDataset.getSamples();
	const auto labels = trainingDataset.getLabels();

	std::vector<uint64_t> svIds;
	std::vector<svmComponents::DatasetVector> svDatasetVectors;

	auto temp = svm;
	auto supportVectors = temp->getSupportVectors();

	for (int i = 0; i < supportVectors.size(); i++)
	{
		const float* sv = supportVectors[i].data();
		constexpr auto epsilon = 1e-3; //0.001;
		auto positionSv = std::find_if(samples.begin(), samples.end(), [&sv, &epsilon](auto& sample)
		{
			return abs(sample[1] - sv[1]) < epsilon && abs(sample[0] - sv[0]) < epsilon;
		}) - samples.begin();

		if (positionSv == samples.size())
		{
			//LOG_F(INFO, "Support vector not found in individual, can happen with adding vectors from frozen Pool of SV into training, using DA-SVM kernels");
			continue;
		}

		svIds.emplace_back(ids[positionSv]);
		svDatasetVectors.emplace_back(ids[positionSv], labels[positionSv]);
	}

	return {svDatasetVectors, svIds};
}

dataset::Dataset<std::vector<float>, float> createDatasetFromIds(const dataset::Dataset<std::vector<float>, float>& dataset,
                                                                 std::vector<uint64_t>& manualDataset)
{
	auto targets = dataset.getLabels();
	std::vector<svmComponents::DatasetVector> chromosomeDataset;
	for (auto i = 0; i < manualDataset.size(); ++i)
	{
		chromosomeDataset.emplace_back(svmComponents::DatasetVector(manualDataset[i], static_cast<std::uint8_t>(targets[manualDataset[i]])));
	}

	auto manual = svmComponents::SvmTrainingSetChromosome(std::move(chromosomeDataset));
	auto shrinkedSet = manual.convertChromosome(dataset);
	return shrinkedSet;
}

std::pair<std::vector<svmComponents::DatasetVector>, std::vector<uint64_t>> getUncertainDataset(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                                                const std::vector<uint64_t>& ids,
                                                                                                std::shared_ptr<phd::svm::ISvm> svm)
{
	auto samples = trainingSet.getSamples();
	auto labels = trainingSet.getLabels();

	const size_t interationCount = samples.size();

	std::vector<svmComponents::DatasetVector> uncertainTrainingSet(interationCount, svmComponents::DatasetVector{ 0, -1000});
	
	
	auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(svm.get());

	#pragma omp parallel for schedule(static)
	for (long long  i = 0; i < static_cast<long long>(interationCount); i++)
	{
		auto result = res->classifyWithCertainty(samples[i]);
		if (result == -100) //if we are not sure we pass this example further into list
		{
			
			{
				//uncertainIds.push_back(ids[i]);
				//uncertainTrainingSet.push_back(svmComponents::DatasetVector{ ids[i], labels[i] });
				uncertainTrainingSet[i] = svmComponents::DatasetVector{ ids[i], labels[i] };
			}
		}
	}
	uncertainTrainingSet.erase(std::remove_if(uncertainTrainingSet.begin(), uncertainTrainingSet.end(),
	                                          [](const DatasetVector& o)
	                                          {
		                                          return o.classValue == -1000;
	                                          }), uncertainTrainingSet.end());
	
	std::vector<uint64_t> uncertainIds;
	uncertainIds.reserve(uncertainTrainingSet.size());
	for(auto& obj : uncertainTrainingSet)
	{
		uncertainIds.emplace_back(obj.id);
	}
	

	return {uncertainTrainingSet, uncertainIds};
}

std::vector<uint64_t> countClasses(const dataset::Dataset<std::vector<float>, float>& trainingDataset, unsigned numberOfClasses)
{
	std::vector<uint64_t> count;
	count.resize(numberOfClasses);
	for (auto& v : trainingDataset.getLabels())
	{
		count[static_cast<int>(v)]++;
	}

	return count;
}

std::vector<uint64_t> countClasses(const std::vector<DatasetVector>& trainingDataset, unsigned numberOfClasses)
{
	std::vector<uint64_t> count;
	count.resize(numberOfClasses);
	for (auto& v : trainingDataset)
	{
		count[static_cast<int>(v.classValue)]++;
	}

	return count;
}

bool is_empty(std::string& filename)
{
	std::ifstream pFile(filename);
	return pFile.peek() == std::ifstream::traits_type::eof();
}
} // namespace genetic
