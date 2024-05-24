
#include <string>
#include <fstream>
#include <iostream>
#include "libPlatform/Subprocess.h"
#include "libPlatform/PlatformException.h"
#include "PythonFeatureSelection.h"
#include "libPlatform/SubtreeExceptions.h"
#include "libPlatform/StringUtils.h"
#include "TestApp/PythonPath.h"

class Error : public platform::PlatformException
{
public:
	explicit Error(std::string message)
		: PlatformException(std::move(message))
	{
	}
};

std::vector<bool> runFeatureSelection(std::filesystem::path treningSetPath)
{
	try
	{
		// filesystem::FileSystem fs;

		// auto pythonScriptPath = filesystem::Path("featureSelection.py");
		// if (!fs.exists(pythonScriptPath))
		// {
		// 	throw platform::FileNotFoundException(pythonScriptPath.string());
		// }

		// const auto command = std::string(PYTHON_PATH + " featureSelection.py -t "
		// 	+ treningSetPath.string());

		// std::cout << "Starting python script\n";

		// const auto[output, ret] = platform::subprocess::launchWithPipe(command);
		// if (ret != 0)
		// {
		// 	//@wdudzik fix this in future
		// 	throw Error("Python, featureSelection.py script failed. Output of script: " + output);
		// }
		// else
		// {
		// 	std::cout << output;
		// }

		// std::ifstream featureSelectionResult(treningSetPath.parent_path().string() + "\\" + "featureSelection.txt", std::fstream::in);
		// std::string features;
		// std::vector<std::string> tokens;
		// std::getline(featureSelectionResult, features);
		// platform::stringUtils::splitString(features, ',', tokens);
		std::vector<bool> featuresT;

		//std::transform(tokens.begin(), tokens.end(), std::back_inserter(featuresT), [](const std::string& string) { return string == "1"; });

		return featuresT;
	}
	catch (const std::runtime_error& e)
	{
		throw;
	}
	catch (...)
	{
		//@wdudzik fix this in future
		throw Error("Something terrible wrong happened during running featureSelection.py script");
	}

}

std::vector<double> runMutualInfo(std::filesystem::path treningSetPath)
{
	try
	{
		// filesystem::FileSystem fs;

		// auto pythonScriptPath = filesystem::Path("mutualInfo.py");
		// if (!fs.exists(pythonScriptPath))
		// {
		// 	throw platform::FileNotFoundException(pythonScriptPath.string());
		// }

		// const auto command = std::string(PYTHON_PATH + " mutualInfo.py -t "
		// 	+ treningSetPath.string());

		// std::cout << "Starting python script\n";

		// const auto[output, ret] = platform::subprocess::launchWithPipe(command);
		// if (ret != 0)
		// {
		// 	//@wdudzik fix this in future
		// 	throw Error("Python, mutualInfo.py script failed. Output of script: " + output);
		// }
		// else
		// {
		// 	std::cout << output;
		// }

		// std::ifstream featureSelectionResult(treningSetPath.parent_path().string() + "\\" + "probabilites_of_features.txt", std::fstream::in);
		// std::string features;
		// std::vector<std::string> tokens;
		// std::getline(featureSelectionResult, features);
		// platform::stringUtils::splitString(features, ',', tokens);
		std::vector<double> mutualInfo;

		// std::transform(tokens.begin(), tokens.end(), std::back_inserter(mutualInfo), [](const std::string& string)
		// {
		// 	return std::stod(string);
		// });

		return mutualInfo;
	}
	catch (const std::runtime_error& /*e*/)
	{
		throw;
	}
	catch (...)
	{
		//@wdudzik fix this in future
		throw Error("Something terrible wrong happened during running featureSelection.py script");
	}
}