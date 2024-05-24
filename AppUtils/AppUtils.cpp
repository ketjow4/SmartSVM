
#include <filesystem>
#include "libSvmComponents/MemeticMutialInfoRoulleteWheelGeneration.h"
#include "libPlatform/TimeUtils.h"
#include "AppUtils.h"
#include "libDataset/CsvReader.h"

#include <cxxopts/cxxopts.hpp>

namespace testApp
{
    std::string getLastLine(std::string folder, std::string filename)
    {
        std::ifstream fin;
        fin.open(folder + "\\" + filename, std::ios_base::in);
        if (fin.is_open())
        {
            std::string lastline;
            while (fin >> std::ws && std::getline(fin, lastline)) // skip empty lines
                ;
            fin.close();
            return lastline;
        }
    }

    void createSummaryFile(platform::Subtree& config, std::ofstream& summaryFile, std::vector<std::string>& logFileNames, Verbosity verbosity)
    {
        for (const auto& logFilename : logFileNames)
        {
            const auto lastLine = testApp::getLastLine(config.getValue<std::string>("Svm.OutputFolderPath"), logFilename);
            summaryFile << lastLine << "\n";
        }
        summaryFile.close();
        if (verbosity != Verbosity::All)
        {
            for (const auto& logFilename : logFileNames)
            {
                std::filesystem::remove(config.getValue<std::string>("Svm.OutputFolderPath") + "\\" + logFilename);
            }
        }
    }

    std::string createOutputFolder(std::string& outputFolderName)
    {
        auto timestamp = timeUtils::getTimestamp();
        outputFolderName.pop_back();
        auto folderName = outputFolderName + "__" + timestamp;

        if (std::filesystem::create_directories(folderName) == false)
        {
            throw std::runtime_error("Error creating output directory");
        }
        return folderName;
    }


    std::string createOutputFolderWithDetails(std::string& outputFolderName, std::string& expDetails)
    {
        //filesystem::FileSystem fs;
        auto timestamp = timeUtils::getTimestamp();
        if (outputFolderName.back() == '\\')
        {
            outputFolderName.pop_back();
        }
        auto folderName = outputFolderName + "__" + timestamp + "__" + expDetails;

        if (std::filesystem::create_directories(folderName) == false)
        {
            throw std::runtime_error("Error creating output directory");
        }
        return folderName;
    }


    std::string getLogFilename(int fold, int i, const std::string& configFile)
    {
        return timeUtils::getTimestamp() + "_" +
            configFile + "_" +
            std::to_string(i) + "fold_" +
            std::to_string(fold) + ".txt";
    }

    void dumpVectorToFile(filesystem::DiskFile& resultFile, const std::vector<float>& minVec)
    {
        std::stringstream vectorAsString;
        std::copy(minVec.begin(), minVec.end(), std::ostream_iterator<float>(vectorAsString, "\t"));
        vectorAsString << "\n";

        resultFile.write(gsl::span<const unsigned char>(reinterpret_cast<const unsigned char*>(vectorAsString.str().c_str()), vectorAsString.str().length()));
    }

    void loadDataAndDumpScalingValues(genetic::SvmWokrflowConfiguration& con, genetic::LocalFileDatasetLoader& dataLoading)
    {
        try
        {
            auto trainingSetPath = con.trainingDataPath;
            const auto filename = trainingSetPath.filename().string();
            const auto scallingFilePath = trainingSetPath.parent_path().append(filename + ".scalling");
            if (!exists(scallingFilePath))
            {
                filesystem::DiskFile resultFile(scallingFilePath, "w");

                const auto minVec = dataLoading.scalingVectorMin();
                const auto maxVec = dataLoading.scalingVectorMax();

                testApp::dumpVectorToFile(resultFile, minVec);
                testApp::dumpVectorToFile(resultFile, maxVec);
            }
        }
        catch (const std::runtime_error& exception)
        {
            std::cout << exception.what();
        }
    }

    void saveSvmModel(platform::Subtree& config, int fold, int i, phd::svm::ISvm& resultModel, const std::string& configFile)
    {
        resultModel.save(config.getValue<std::string>("Svm.OutputFolderPath") +
            timeUtils::getTimestamp() + "_" +
            configFile + "_" +
            std::to_string(i) + "fold_" +
            std::to_string(fold) + "_svmModel.xml");
    }

    void saveSvmResultsToFile(platform::Subtree& config, int fold, int i, phd::svm::ISvm& resultModel, const std::string& configFile,
        genetic::IDatasetLoader& loadingWorkflow)
    {
        std::ofstream outputFile{ config.getValue<std::string>("Svm.OutputFolderPath") +
	        timeUtils::getTimestamp() + "_" + configFile + "_" + std::to_string(i) + "fold_" +
            std::to_string(fold) + "_svmModel.SvmResults", std::ios_base::out };

        if (outputFile.is_open())
        {
            auto featureSet = resultModel.getFeatureSet();
            auto featureChromosome = svmComponents::SvmFeatureSetMemeticChromosome(std::move(featureSet));
            auto training = featureChromosome.convertChromosome(loadingWorkflow.getTraningSet());
            auto validation = featureChromosome.convertChromosome(loadingWorkflow.getValidationSet());
            auto test = featureChromosome.convertChromosome(loadingWorkflow.getTestSet());

            outputFile << "#Training set" << "\n";

            for (auto& vec : training.getSamples())
            {
                outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
            }

            outputFile << "#Validation set" << "\n";
            for (auto& vec : validation.getSamples())
            {
                outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
            }

            outputFile << "#Test set" << "\n";
            for (auto& vec : test.getSamples())
            {
                outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
            }
        }

        outputFile.close();
    }

    void saveSvmResultsToFile2(std::string outputfolder, int i, phd::svm::ISvm& resultModel,
        genetic::IDatasetLoader& loadingWorkflow, bool useFeateures)
    {
	    std::ofstream outputFile{
		    outputfolder +
		    timeUtils::getTimestamp() + "_" + "_" + std::to_string(i) +
		    "_svmModel.SvmResults",
		    std::ios_base::out
	    };

	    if (outputFile.is_open())
	    {
            auto training = (loadingWorkflow.getTraningSet());
            auto validation = (loadingWorkflow.getValidationSet());
            auto test = (loadingWorkflow.getTestSet());
             
            if(useFeateures)
            {
	            auto featureSet = resultModel.getFeatureSet();
            	auto featureChromosome = svmComponents::SvmFeatureSetMemeticChromosome(std::move(featureSet));
            	training = featureChromosome.convertChromosome(loadingWorkflow.getTraningSet());
            	validation = featureChromosome.convertChromosome(loadingWorkflow.getValidationSet());
            	test = featureChromosome.convertChromosome(loadingWorkflow.getTestSet());
            }
		    outputFile << "#Training set" << "\n";

		    for (auto& vec : training.getSamples())
		    {
			    outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
		    }

		    outputFile << "#Validation set" << "\n";
		    for (auto& vec : validation.getSamples())
		    {
			    outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
		    }

		    outputFile << "#Test set" << "\n";
		    for (auto& vec : test.getSamples())
		    {
			    outputFile << resultModel.classifyHyperplaneDistance(vec) << "\n";
		    }
	    }

	    outputFile.close();
    }


    void saveSvmResultsToFile3(std::string outputfolder, int i, phd::svm::ISvm& resultModel,
        genetic::IDatasetLoader& loadingWorkflow)
    {
        std::ofstream outputFile{
            outputfolder +
           "training__" + std::to_string(i) +
            "_svmModel.SvmResults",
            std::ios_base::out
        };

        std::ofstream outputFile_val{
            outputfolder +
           "validation__" + std::to_string(i) +
            "_svmModel.SvmResults",
            std::ios_base::out
        };

        std::ofstream outputFile_test{
            outputfolder +
           "test__" + std::to_string(i) +
            "_svmModel.SvmResults",
            std::ios_base::out
        };

        if (outputFile.is_open())
        {
            auto training = (loadingWorkflow.getTraningSet());
            auto validation = (loadingWorkflow.getValidationSet());
            auto test = (loadingWorkflow.getTestSet());

            outputFile << "#Training set" << "\n";

            auto labels = training.getLabels();
            auto samples = training.getSamples();
            for (auto i = 0u; i < samples.size(); ++i)
            {
                outputFile << resultModel.classifyHyperplaneDistance(samples[i]) << "," << labels[i]   << "\n";
            }

            outputFile_val << "#Validation set" << "\n";
            labels = validation.getLabels();
            samples = validation.getSamples();
            for (auto i = 0u; i < samples.size(); ++i)
            {
                outputFile_val << resultModel.classifyHyperplaneDistance(samples[i]) << "," << labels[i]  << "\n";
            }

            outputFile_test << "#Test set" << "\n";
            labels = test.getLabels();
            samples = test.getSamples();
            for (auto i = 0u; i < samples.size(); ++i)
            {
                outputFile_test << resultModel.classifyHyperplaneDistance(samples[i]) << "," << labels[i] << "\n";
            }
        }

        outputFile.close();
        outputFile_val.close();
        outputFile_test.close();
    }

    void SaveGroupsResult(phd::svm::ISvm& resultModel,
                          std::ofstream& outputFile,
                          const dataset::Dataset<std::vector<float>, float>& training)
    {
	    outputFile << "#Raw score, label, group" << "\n";
	    auto labels = training.getLabels();
	    auto samples = training.getSamples();
	    auto groups = training.getGroups();
	    for (auto i = 0u; i < samples.size(); ++i)
	    {
		    outputFile << resultModel.classifyHyperplaneDistance(samples[i]) << "," << labels[i] << "," << groups[i] << "\n";
	    }
	    auto raw_scores = resultModel.classifyGroupsRawScores(training);
	    auto scores = resultModel.classifyGroups(training);
	    outputFile << "#Groups answers\n";
	    outputFile << "#Group, raw score after propagation, classify result\n";
	    for (auto i = 0u; i < raw_scores.size(); ++i)
	    {
		    outputFile << i << "," << raw_scores[i] << "," << scores[i] << "\n";
	    }
    }

    void saveSvmGroupsResultsToFile(platform::Subtree& config, int /*fold*/, int i, phd::svm::ISvm& resultModel, const std::string& /*configFile*/,
                                    genetic::IDatasetLoader& loadingWorkflow)
    {
        auto outputfolder = config.getValue<std::string>("Svm.OutputFolderPath");

        std::ofstream outputFile{
         outputfolder +
        "training__" + std::to_string(i) +
         "_svmModel.SvmResults",
         std::ios_base::out
        };

        std::ofstream outputFile_val{
            outputfolder +
           "validation__" + std::to_string(i) +
            "_svmModel.SvmResults",
            std::ios_base::out
        };

        std::ofstream outputFile_test{
            outputfolder +
           "test__" + std::to_string(i) +
            "_svmModel.SvmResults",
            std::ios_base::out
        };

        if (outputFile.is_open())
        {
            auto training = (loadingWorkflow.getTraningSet());
            auto validation = (loadingWorkflow.getValidationSet());
            auto test = (loadingWorkflow.getTestSet());


            SaveGroupsResult(resultModel, outputFile, training);
            SaveGroupsResult(resultModel, outputFile_val, validation);
            SaveGroupsResult(resultModel, outputFile_test, test);
        }

        outputFile.close();
        outputFile_val.close();
        outputFile_test.close();
    }

    void saveRepositoryState(const std::filesystem::path& outPath)
    {
        try
        {
            auto pythonScriptPath = std::filesystem::path("save.py");
            if (!std::filesystem::exists(pythonScriptPath))
            {
                throw platform::FileNotFoundException(pythonScriptPath.string());
            }

            const auto command = std::string(PYTHON_PATH + " save.py -output_path " + outPath.string());

            std::cout << "Starting python script\n";

            const auto [output, ret] = platform::subprocess::launchWithPipe(command);
            if (ret != 0)
            {
                //@wdudzik fix this in future
                std::cout << "Python, save.py script failed. Output of script: " + output;
                //throw std::runtime_error("Python, save.py script failed. Output of script: " + output);
            }
            else
            {
                std::cout << output;
            }
        }
        catch (const std::runtime_error& )
        {
            throw;
        }
        catch (...)
        {
            //@wdudzik fix this in future
            std::cout << "Something terrible wrong happened during running save.py script";
            //throw Error("Something terrible wrong happened during running featureSelection.py script");
        }

    }



    configTestApp parseCommandLineArguments(int argc, char** argv)
    {
        configTestApp config;
        cxxopts::Options options("SmartSVM", "Allowed options");

        options.add_options()
        //("c,configFile", "json config file to load", cxxopts::value<std::string>()) 
        ("d,datafolder", "path to the folder in which dataset files are stored\n", cxxopts::value<std::string>())
        ("o,outputfolder", "path to the folder in which results will be stored (on default the same as datafolder", cxxopts::value<std::string>())
        //("v,verbosity", "Level of verbosity of program (affects the number of files created with saved results)", cxxopts::value<testApp::Verbosity>()->default_value(Verbosity::Standard))
        ;

        auto result = options.parse(argc, argv);

   
        //config.configFile = result["configFile"].as<std::string>();
        config.datafolder = result["datafolder"].as<std::string>();
        config.outputFolder = result["outputfolder"].as<std::string>();
        //config.verbosity = result["verbosity"].as<testApp::Verbosity>();
        config.verbosity = testApp::Verbosity::All;

        if (config.outputFolder.empty())
        {
            config.outputFolder = config.datafolder;
        }
        

        // using namespace boost::program_options;
        // options_description desc("Allowed options");
        // desc.add_options()
        //     ("help,h", "print usage message\n")
        //     //("configFile,c", value(&config.configFile), "json config file to load")
        //     ("datafolder,d", value<std::string>(&config.datafolder), "path to the folder in which dataset files are stored\n")
        //     ("outputfolder,o", value<std::string>(&config.outputFolder)->default_value(""), "path to the folder in which results will be stored (on default the same as datafolder)")
        //     ("verbosity,v", value<testApp::Verbosity>(&config.verbosity)->default_value(Verbosity::Standard), "Level of verbosity of program (affects the number of files created with saved results)");

        // variables_map vm;
        try
        {
        //     store(parse_command_line(argc, argv, desc), vm);
        //     if (vm.count("help"))
        //     {
        //         std::cout << "Basic Command Line Parameter App" << std::endl << desc << std::endl;

        //         return config;
        //     }
        //     notify(vm);

        //     if (config.outputFolder.empty())
        //     {
        //         config.outputFolder = config.datafolder;
        //     }

            return config;
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl << std::endl;
            //std::cout << desc << std::endl;
            return config;
        }
    }

    std::vector<std::filesystem::path> getAllConfigFiles(std::filesystem::path folderPath)
    {
        std::vector<std::filesystem::path> configFiles;
        for (auto& file : std::filesystem::directory_iterator(folderPath))
        {
            if (file.path().extension().string() == ".json")
            {
                configFiles.push_back(file.path());
            }
        }
        std::sort(configFiles.begin(), configFiles.end());
        return configFiles;
    }

    std::vector<std::string> listDirectories(const std::filesystem::path& path)
    {
        std::vector<std::string> dirs;
        for (const auto& dirIt : std::filesystem::directory_iterator(path))
        {
           
            if (std::filesystem::is_directory(dirIt.path()))
            {
                dirs.emplace_back(dirIt.path().string());
            }
        }
        if (dirs.empty())
        {
            std::cout << "No folder found in " + path.string();
        }
        return dirs;
    }

    std::vector<unsigned> countLabels2(unsigned numberOfClasses, const dataset::Dataset<std::vector<float>, float>& dataset)
    {
        std::vector<unsigned int> labelsCount(numberOfClasses);
        auto targets = dataset.getLabels();
        std::for_each(targets.begin(), targets.end(),
            [&labelsCount](const auto& label)
            {
                ++labelsCount[static_cast<int>(label)];
            });
        return labelsCount;
    }

    DatasetInfo getInfoAboutDataset(const std::filesystem::path& path)
    {
        dataset::Dataset<std::vector<float>, float> dataset;
        if(path.extension() == ".csv")
        {
	        dataset = phd::data::readCsv(path);
        }
        else
        {
            dataset = phd::data::readCsvGroups(path);
        }

        auto numberOfFeatures = dataset.getSample(0).size();
        auto datasetSize = dataset.size();


        auto labels = dataset.getLabels();
        auto numberOfClasses = std::set<float>(labels.begin(), labels.end()).size();

        std::vector<uint32_t> kValues;
        const auto maxKValue = 512;
        auto classCount = countLabels2(numberOfClasses, dataset);
        const auto numberOfExamplesFromSmallerClass = *std::min_element(classCount.begin(), classCount.end());

        for (auto i = numberOfFeatures; i < maxKValue && i < numberOfExamplesFromSmallerClass; i *= 4)
        {
            kValues.emplace_back(static_cast<uint32_t>(i));
        }

        return DatasetInfo(datasetSize, numberOfFeatures, kValues, numberOfClasses);
    }
}
