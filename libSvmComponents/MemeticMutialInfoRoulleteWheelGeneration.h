#pragma once

#include <memory>
#include <filesystem>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/Population.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "SvmFeatureSetMemeticChromosome.h"
#include "libPlatform/SubtreeExceptions.h"
#include "libPlatform/Subprocess.h"
#include "libPlatform/StringUtils.h"
#include <fstream>
#include "SvmComponentsExceptions.h"
#include "SmallerPoolExperiment.h"

#include "TestApp/PythonPath.h"

namespace svmComponents
{
class Error : public platform::PlatformException
{
public:
    explicit Error(std::string message)
        : PlatformException(std::move(message))
    {
    }
};

inline void saveDataset24433(const dataset::Dataset<std::vector<float>, float>& set, std::string name)
{
    std::ofstream output(name);

    for (auto i = 0u; i < set.getSamples().size(); ++i)
    {
        auto sample = set.getSample(i);
        for (auto& feature : sample)
        {
            output << feature << ",";
        }
        output << set.getLabel(i) << "\n";
    }
    output.close();
}

inline std::vector<double> runMutualInfo(std::filesystem::path treningSetPath, std::filesystem::path outputPath, const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/)
{
    try
    {

        auto pythonScriptPath = std::filesystem::path("mutualInfo.py");
        if (!std::filesystem::exists(pythonScriptPath))
        {
            throw platform::FileNotFoundException(pythonScriptPath.string());
        }

       /* auto tr_path = outputPath.parent_path().string() + "\\" + "trainingSet_FS.csv";
        saveDataset24433(trainingSet, tr_path);*/

    	//if file exists this mean that feature algorithm was already run 
        if (!std::filesystem::exists(outputPath.parent_path().string() + "\\" + "probabilites_of_features.txt"))
        {
	        const auto command = std::string(PYTHON_PATH + " mutualInfo.py -t "
	            + treningSetPath.string() + " -o " + outputPath.string());

            /*const auto command = std::string(PYTHON_PATH + " mutualInfo.py -t "
                + tr_path + " -o " + outputPath.string());*/

	        std::cout << "Starting python script\n";

	        const auto [output, ret] = platform::subprocess::launchWithPipe(command);
	        if (ret != 0)
	        {
	            //@wdudzik fix this in future
	            throw Error("Python, mutualInfo.py script failed. Output of script: " + output);
	        }
	        else
	        {
	            std::cout << output;
	        }
        }

        	
        std::ifstream featureSelectionResult(outputPath.parent_path().string() + "\\" + "probabilites_of_features.txt", std::fstream::in);
        std::string features;
        std::vector<std::string> tokens;
        std::getline(featureSelectionResult, features);
        platform::stringUtils::splitString(features, ',', tokens);
        std::vector<double> mutualInfo;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(mutualInfo), [](const std::string& string)
        {
            return std::stod(string);
        });

        return mutualInfo;
    }
    catch (const std::exception& /*e*/)
    {
        throw;
    }
    catch (...)
    {
        //@wdudzik fix this in future
        throw Error("Something terrible wrong happened during running featureSelection.py script");
    }
}

class MemeticMutialInfoRoulleteWheelGeneration : public geneticComponents::IPopulationGeneration<SvmFeatureSetMemeticChromosome>
{
public:
    explicit MemeticMutialInfoRoulleteWheelGeneration(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                      std::unique_ptr<random::IRandomNumberGenerator> rngEngine,
                                                      unsigned int numberOfClassExamples,
                                                      std::string trainingDataPath,
                                                      std::string outputPath)
        : m_trainingSet(trainingSet)
        , m_rngEngine(std::move(rngEngine))
        , m_numberOfClassExamples(numberOfClassExamples)
        , m_trainingDataPath(std::move(trainingDataPath))
		, m_outputPath(std::move(outputPath))
    {
        if (m_numberOfClassExamples > trainingSet.getSample(0).size())
        {
            throw ValueOfClassExamplesIsTooHighForDataset(m_numberOfClassExamples);
        }
    }

    std::uint64_t getFeatureId(const std::vector<double>& mutualInfo, double random) const
    {
        for (auto i = 0u; i < mutualInfo.size(); ++i)
        {
            random -= mutualInfo[i];
            if (random < 0)
            {
                return i;
            }
        }
        return mutualInfo.size() - 1;
    }

    geneticComponents::Population<SvmFeatureSetMemeticChromosome> createPopulation(uint32_t populationSize) override
    {
        if (populationSize == 0)
        {
            throw geneticComponents::PopulationIsEmptyException();
        }

        //auto mutualInfo = runMutualInfo(m_trainingDataPath, m_outputPath, m_trainingSet);
        auto features = m_trainingSet.getSample(0).size();
        auto mutualInfo = std::vector<double>(features, 1.0/features);

        auto mask = toMask(mutualInfo);
        SmallerPool::instance().setupMask(mask);

        auto random01 = std::uniform_real_distribution<double>(0.0, 1.0);
        std::vector<SvmFeatureSetMemeticChromosome> population(populationSize);

        std::generate(population.begin(), population.end(), [&]
        {
            std::unordered_set<std::uint64_t> trainingSet;
            std::vector<Feature> chromosomeDataset;
            chromosomeDataset.reserve(m_numberOfClassExamples);
            unsigned int featureCount = 0;
            int maxTries = 1000000;
            int tries = 0;
        	
            while (featureCount != m_numberOfClassExamples && tries < maxTries)
            {
                auto randomValue = m_rngEngine->getRandom(random01);

                auto featureId = getFeatureId(mutualInfo, randomValue);

                if (featureCount < m_numberOfClassExamples && // less that desired number of class examples
                    trainingSet.emplace(featureId).second) // is unique
                {
                    chromosomeDataset.emplace_back(Feature(featureId));
                    featureCount++;
                }

                tries++;
            }
            return SvmFeatureSetMemeticChromosome(std::move(chromosomeDataset));
        });
        return geneticComponents::Population<SvmFeatureSetMemeticChromosome>(std::move(population));
    }

private:
    const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
    std::unique_ptr<random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClassExamples;
    std::string m_trainingDataPath;
    std::string m_outputPath;
};
} // namespace svmComponents
