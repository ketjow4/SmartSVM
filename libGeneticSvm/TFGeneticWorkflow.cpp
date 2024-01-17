
#pragma once

#include "libGeneticSvm/TFGeneticWorkflow.h"
#include "libPlatform/StringUtils.h"
#include <sstream>
#include <iterator>


namespace genetic
{
TFGeneticWorkflow::TFGeneticWorkflow(const SvmWokrflowConfiguration& config, TFGeneticEvolutionConfiguration algorithmConfig)
    : m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
    , m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(std::move(algorithmConfig))
{
}

std::shared_ptr<phd::svm::ISvm> TFGeneticWorkflow::run()
{
	std::vector<svmComponents::BaseSvmChromosome> classifier;
	std::vector<std::string> logentries;

	for (int i = 0; i < 5; i++)
	{

		m_trainingSetOptimization->initialize();


		//static unsigned int featureNumber = static_cast<unsigned>(m_trainingSetOptimization->getBestTrainingSet().getSample(0).size());

		//m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
		//m_featureSetOptimization->initialize();

		log(*m_trainingSetOptimization);
		//log(*m_featureSetOptimization);

		while (true)
		{
			//m_trainingSetOptimization->setupFeaturesSet(convertToOldChromosome(m_featureSetOptimization->getBestChromosomeInGeneration(), featureNumber));

			m_trainingSetOptimization->runGeneticAlgorithm();
			log(*m_trainingSetOptimization);

			if (isFinished())
			{
				classifier.push_back(m_trainingSetOptimization->getBestChromosomeInGeneration());
				logentries.push_back(*m_trainingSetOptimization->getResultLogger().getLogEntries().crbegin());

				break;
			}

			clearlog(*m_trainingSetOptimization);
			//m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());

			//m_featureSetOptimization->runGeneticAlgorithm();
			//log(*m_featureSetOptimization);

			/*if (isFinished())
			{
				return (m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier());
			}*/
		}
	}

	auto bestOne = std::max_element(classifier.begin(), classifier.end(),
		[](svmComponents::BaseSvmChromosome left, svmComponents::BaseSvmChromosome right)
	{
		return left.getFitness() < right.getFitness();
	});
	auto it = std::find_if(classifier.begin(), classifier.end(),
		[&bestOne](svmComponents::BaseSvmChromosome element)
	{
		return element.getFitness() == bestOne->getFitness();
	});
	auto pos = std::distance(classifier.begin(), it);
	std::vector<std::string> a;


	auto finalEntry = ::platform::stringUtils::splitString(logentries[pos], '\t');

	auto time = ::platform::stringUtils::splitString(*logentries.rbegin(), '\t')[3];

	finalEntry[3] = time;

	std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
		[](std::string &ss, std::string &s)
	{
		return ss.empty() ? s : ss + "\t" + s;
	});
	//s.append("\n");

	a.push_back(s);

	m_trainingSetOptimization->getResultLogger().setEntries(a);
	m_trainingSetOptimization->getResultLogger().logToFile(m_resultFilePath);

	return bestOne->getClassifier();
}

bool TFGeneticWorkflow::isFinished() const
{
	return m_algorithmConfig.m_stopTrainingSet->isFinished(m_trainingSetOptimization->getPopulation()); //&&
            //m_algorithmConfig.m_stopFeatureSet->isFinished(m_featureSetOptimization->getPopulation());
}
} // namespace genetic
