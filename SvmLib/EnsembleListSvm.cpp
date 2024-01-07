#include "EnsembleListSvm.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "libPlatform/loguru.hpp"

namespace phd { namespace svm

{
EnsembleListSvm::EnsembleListSvm(std::shared_ptr<phd::svm::ListNodeSvm> list, int length)
	: root(list)
	, list_length(length)
{
}

EnsembleListSvm::EnsembleListSvm(std::shared_ptr<phd::svm::ListNodeSvm> list, int length, bool newClassification)
	: root(list)
	, list_length(length)
	, newClassificationScheme(newClassification)
{
}

EnsembleListSvm::EnsembleListSvm(const std::filesystem::path& filepath, bool newClassification)
{
	list_length = 0;
	newClassificationScheme = newClassification;
	//LOG_F(INFO, "change default value for classification scheme after experiment");
	
	std::ifstream input(filepath);

	if(input.is_open())
	{
		std::stringstream buffer;
		buffer << input.rdbuf();
		auto model_text = buffer.str();

		auto temp = std::make_shared<ListNodeSvm>();
		root = temp;
		temp->m_next = nullptr;
		
		auto global_pos = 0;
		while(true)
		{
			auto pos = model_text.find("#Node", global_pos);

			if (pos == std::string::npos)
				break;

			size_t new_line_after_node = model_text.find('\n', pos);
			size_t second_node = model_text.find("#Node", new_line_after_node);

			if (second_node == std::string::npos)
			{
				second_node = model_text.length();
			}

			auto text = model_text.substr(new_line_after_node + 1, second_node - new_line_after_node - 1); // +1 to move past \n symbol
			//std::cout << text;

			auto model = std::make_shared<libSvmImplementation>();
			model->loadFromString(text);
			temp->m_svm = model;
			
			
			global_pos += static_cast<int>(second_node - new_line_after_node);
			list_length += 1;

			auto pos2 = model_text.find("#Node", global_pos); //looking if there is another svm in the ensemble

			if (pos2 == std::string::npos)
				break;

			temp->m_next = std::make_shared<ListNodeSvm>(); //if we arrive here need to create a new node for next svm
			temp = temp->m_next;
			temp->m_next = nullptr;
		}
	}
}


//THIS IS especially for loading other (SE-SVM) trained model as last node which handles uncertain region
EnsembleListSvm::EnsembleListSvm(const std::filesystem::path& filepath, bool newClassification, bool /*SESVM_AS_LAST*/)
{
	list_length = 0;
	newClassificationScheme = newClassification;
	//LOG_F(INFO, "change default value for classification scheme after experiment");

	std::ifstream input(filepath);

	if (input.is_open())
	{
		std::stringstream buffer;
		buffer << input.rdbuf();
		auto model_text = buffer.str();

		auto temp = std::make_shared<ListNodeSvm>();
		root = temp;
		temp->m_next = nullptr;

		auto global_pos = 0;
		while (true)
		{
			auto pos = model_text.find("#Node", global_pos);

			if (pos == std::string::npos)
				break;

			size_t new_line_after_node = model_text.find('\n', pos);
			size_t second_node = model_text.find("#Node", new_line_after_node);

			if (second_node == std::string::npos)
			{
				second_node = model_text.length();
			}

			auto text = model_text.substr(new_line_after_node + 1, second_node - new_line_after_node - 1); // +1 to move past \n symbol
			//std::cout << text;

			auto model = std::make_shared<libSvmImplementation>();
			model->loadFromString(text);
			temp->m_svm = model;


			global_pos += static_cast<int>(second_node - new_line_after_node);
			list_length += 1;

			auto pos2 = model_text.find("#Node", global_pos); //looking if there is another svm in the ensemble

			//in here we now that this was last node so we shoul load SE-SVM on its place
			if (pos2 == std::string::npos)
			{
				//filesystem::FileSystem fs;
				//auto mainFolder = filepath.parent_path() / "..\\";
				//
				//for (auto& p : std::filesystem::directory_iterator(mainFolder))
				//{ 
				//	//if (p.is_directory() && (p.path().string().find("SE-SVM") != std::string::npos))
				//	//if (p.is_directory() && (p.path().string().find("ESVM") != std::string::npos))
				//	if (p.is_directory() && (p.path().string().find("GridSearch") != std::string::npos))
				//	{
				//		for (auto& file : std::filesystem::recursive_directory_iterator(p))
				//		{
				//			if (file.path().extension().string() == ".xml")
				//			{
				//				auto svmPath = file.path();
				//				auto model2 = std::make_shared<libSvmImplementation>(svmPath);
				//				temp->m_svm = model2;
				//			}
				//		}
				//	}
				//}
				//
				
				//do not load last node of ALMA or anthting else
				
					if (root != NULL) {

						//1. if head in not null and next of head
						//   is null, release the head
						if (root->m_next == NULL) {
							//root = NULL;
							break;
						}
						else {

							//2. Else, traverse to the second last 
							//   element of the list
							auto temp2 = root;
							while (temp2->m_next->m_next != NULL)
								temp2 = temp2->m_next;

							//3. Change the next of the second 
							//   last node to null and delete the
							//   last node
							auto lastNode = temp2->m_next;
							temp2->m_next = NULL;
						}
					}
				

				
				break;
			}

			temp->m_next = std::make_shared<ListNodeSvm>(); //if we arrive here need to create a new node for next svm
			temp = temp->m_next;
			temp->m_next = nullptr;
		}
	}
}

phd::svm::KernelTypes EnsembleListSvm::getKernelType() const
{
	return phd::svm::KernelTypes::Custom;
}

void EnsembleListSvm::save(const std::filesystem::path& filepath)
{
	std::ofstream output(filepath);

	// if (m_treeEndNode)
	// {
	// 	m_treeEndNode->save(filepath.string() + "ExtraTree.pkl");
	// }
	
	auto temp = root;
	int i = 0;
	while (temp->m_next)
	{
		auto svmPtr = reinterpret_cast<libSvmImplementation*>(temp->m_svm.get());
		auto modelText = svmPtr->saveToString();

		output << "#Node" << i << "\n";
		output << modelText;
		output << "\n";
		i++;
		temp = temp->m_next;
	}

	auto svmPtr = reinterpret_cast<libSvmImplementation*>(temp->m_svm.get());
	auto modelText = svmPtr->saveToString();

	output << "#Node " << i << "\n";
	output << modelText;
	output << "\n";
	
	output.close();
}

bool EnsembleListSvm::isTrained() const
{
	return root != nullptr;
}

bool EnsembleListSvm::canGiveProbabilityOutput() const
{
	return false;
}

bool EnsembleListSvm::canClassifyWithOptimalThreshold() const
{
	return false;
}

uint32_t EnsembleListSvm::getNumberOfKernelParameters(phd::svm::KernelTypes) const
{
	return 0;
}

uint32_t EnsembleListSvm::getNumberOfSupportVectors() const
{
	if (root == nullptr)
		return 0;

	uint32_t sum = 0;

	auto temp = root;
	while (temp->m_next)
	{
		sum += temp->m_svm->getNumberOfSupportVectors();
		temp = temp->m_next;
	}
	sum += temp->m_svm->getNumberOfSupportVectors();

	return sum;
}

std::vector<uint32_t> EnsembleListSvm::getNodesNumberOfSupportVectors() const
{
	if (root == nullptr)
		throw std::exception("No root pointer, cannot create vector with SV number");

	std::vector<uint32_t> result;

	auto temp = root;
	while (temp->m_next)
	{
		result.emplace_back(temp->m_svm->getNumberOfSupportVectors());
		temp = temp->m_next;
	}
	result.emplace_back(temp->m_svm->getNumberOfSupportVectors());

	return result;
}

std::vector<std::vector<float>> EnsembleListSvm::getSupportVectorsOfLastNode()
{
	std::vector<std::vector<float>> svs;
	auto temp = root;

	while (temp->m_next)
	{
		temp = temp->m_next;
	}
	auto svNode = temp->m_svm->getSupportVectors();
	//svs.push_back(svNode);
	svs.insert(svs.end(), svNode.begin(), svNode.end());

	return svs;
}

std::vector<std::vector<float>> EnsembleListSvm::getSupportVectors() const
{
	std::vector<std::vector<float>> svs;
	auto temp = root;

	while (temp->m_next) //while we are not certain
	{
		auto svNode = temp->m_svm->getSupportVectors();
		svs.insert(svs.end(), svNode.begin(), svNode.end());
		temp = temp->m_next;
	}
	auto svNode = temp->m_svm->getSupportVectors();
	svs.insert(svs.end(), svNode.begin(), svNode.end());

	return svs;
}

void EnsembleListSvm::unceratinClosestDistance(float& result, std::vector<ResultAndThresholds> history) const
{
	auto minDistance = 1000000000.0;
	for (auto i = 0u; i < history.size(); ++i)
	{
		auto positiveDistance = std::abs(history[i].positiveThreshold - history[i].results);
		auto negativeDistance = std::abs(history[i].negativeThreshold - history[i].results);

		if (negativeDistance < minDistance && negativeDistance < positiveDistance)
		{
			result = 0;
			minDistance = negativeDistance;
		}
		else if (positiveDistance < minDistance)
		{
			minDistance = positiveDistance;
			result = 1;
		}
	}
}


void EnsembleListSvm::unceratinLargestDistance(float& result, std::vector<ResultAndThresholds> history) const
{
	auto maxDistance = 0.0;
	for (auto i = 0u; i < history.size(); ++i)
	{
		auto positiveDistance = std::abs(history[i].positiveThreshold - history[i].results);
		auto negativeDistance = std::abs(history[i].negativeThreshold - history[i].results);

		if(history[i].positiveThreshold == 1000000)
		{
			positiveDistance = 0;
		}
		if (history[i].negativeThreshold == -1000000)
		{
			negativeDistance = 0;
		}

		if (history[i].results >= 0)
		{
			negativeDistance = 0;
		}
		else if (history[i].results < 0)
		{
			positiveDistance = 0;
		}

		if (negativeDistance > maxDistance && negativeDistance > positiveDistance)
		{
			result = 0;
			maxDistance = negativeDistance;
		}
		else if (positiveDistance > maxDistance)
		{
			maxDistance = positiveDistance;
			result = 1;
		}
	}
}

void EnsembleListSvm::unceratinVoting(float& result, std::vector<ResultAndThresholds> history) const
{
	std::vector<int> voting;
	voting.resize(2);
	for (auto i = 0u; i < history.size(); ++i)
	{
		if (history[i].results < 0)
		{
			voting[0]++;
		}
		else
		{
			voting[1]++;
		}
	}
	if (voting[1] > voting[0])
		result = 1;
	else
		result = 0;
}

std::pair<float, int> EnsembleListSvm::NewScheme(const gsl::span<const float> sample) const
{
	auto temp = root;

	float result = -50;
	std::vector<ResultAndThresholds> history;
	auto j = 0;
	while (temp)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());
		result = res->classifyWithCertainty(sample);
		history.emplace_back(res->classifyDistanceToClosestSV(sample), res->getPositiveCertainty(), res->getNegativeCertainty());		
		
		if (result != -100)
		{
			return std::make_pair(result, j);
			//break;
		}
			

		temp = temp->m_next;
		j++;
	}

		
	//search history for closest SV from list (note which node in list this is)
	auto nodeId = std::distance(history.begin(),
	                            std::max_element(history.begin(), history.end(), 
	                                             [](ResultAndThresholds& a, ResultAndThresholds& b) {return a.results < b.results; }));

	//std::cout << nodeId << "\n";

	//return result based on this node with the closest SV
	temp = root;
	for(auto i = 0; i < nodeId; i++)
	{
		temp = temp->m_next;
	}

	/*auto svmAnswer = temp->m_svm->classifyWithCertainty(sample);
	if(svmAnswer != -100)
	{
		return std::make_pair(svmAnswer, static_cast<int>(nodeId));
	}
	else*/
	{
		auto hyperplaneDistance = temp->m_svm->classifyHyperplaneDistance(sample);
		if (history[nodeId].positiveThreshold - hyperplaneDistance < hyperplaneDistance - history[nodeId].negativeThreshold)
		{
			return std::make_pair(1.0f, static_cast<int>(nodeId));
		}
		else
		{
			return std::make_pair(0.0f, static_cast<int>(nodeId));
		}
	}
	//return std::make_pair(temp->m_svm->classifyWithCertainty(sample), static_cast<int>(nodeId));
}

float EnsembleListSvm::classify(const gsl::span<const float> sample) const
{
	if(newClassificationScheme == false)
	{
		//auto temp = root;

		//float result = -50;
		//std::vector<ResultAndThresholds> history;
		//while (temp)
		//{
		//	auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());
		//	result = res->classifyWithCertainty(sample);
		//	history.emplace_back(res->classifyHyperplaneDistance(sample), res->getPositiveCertainty(), res->getNegativeCertainty());

		//	if (result != -100)
		//		break;

		//	temp = temp->m_next;
		//}

		//if (result == -100)
		//{
		//	//unceratinLargestDistance(result, history); // TODO check in future
		//	unceratinClosestDistance(result, history);

		//	return result;
		//}
		//return result;


		//Upper one is the first apporach,
		return NewScheme(sample).first;
	}
	//NEW CLASSIFICATION SCHEME HERE
	else
	{
		//return DasvmScheme(sample);
		return LastNodeScheme(sample);
		//return NewScheme(sample).first;
	}
}

float EnsembleListSvm::classifyAll(const gsl::span<const float> sample) const
{
	auto temp = root;

	float result = 0;
	while (temp)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());

		result += res->classifyHyperplaneDistance(sample);
		temp = temp->m_next;
	}
	return result;
}

float EnsembleListSvm::DasvmScheme(const gsl::span<const float> sample) const
{
	auto temp = root;

	float result = -50;
	std::vector<ResultAndThresholds> history;
	while (temp)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());
		result = res->classifyWithCertainty(sample);
		history.emplace_back(res->classifyHyperplaneDistance(sample), res->getPositiveCertainty(), res->getNegativeCertainty());

		if (result != -100)
			break;

		if (temp->m_next == nullptr)
		{
			return result;
		}
		temp = temp->m_next;
	}
	return result;
}


float EnsembleListSvm::LastNodeScheme(const gsl::span<const float> sample) const
{
	auto temp = root;

	float result = -50;
	while (temp)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());

		result = res->classifyWithCertainty(sample);
		
		if (temp->m_next == nullptr && result == -100)
		{
			
			// if(m_treeEndNode)
			// {
			// 	return m_treeEndNode->predict(sample);
			// }
			//else
			{
			//	return NewScheme(sample).first;
			}

			
			return static_cast<float>(res->classifyWithOptimalThreshold(sample)); //last node of ALMA, DASVM or other evo-algorithm with full SVM
		}
		
		

		if (result != -100)
			break;


		temp = temp->m_next;
	}
	return result;
}

#pragma optimize("", on)


std::pair<float,int> EnsembleListSvm::LastNodeSchemeAndNode(const gsl::span<const float> sample) const
{
	auto temp = root;

	float result = -50;
	int nodeId = 0;
	while (temp)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());

		result = res->classifyWithCertainty(sample);

		if (temp->m_next == nullptr && result == -100)
		{
			return { result, nodeId };
			
			/*if(m_treeEndNode)
			{
				return m_treeEndNode->predict(sample);
			}
			else
			{
				return NewScheme(sample).first;
			}*/


			//return static_cast<float>(res->classifyWithOptimalThreshold(sample)); //last node of ALMA, DASVM or other evo-algorithm with full SVM
		}



		if (result != -100)
			break;


		temp = temp->m_next;
		nodeId++;
	}
	return { result, nodeId };
}

#pragma optimize("", off)


std::pair<float, int> EnsembleListSvm::classifyWithNode(const gsl::span<const float> sample) const
{
	//return std::pair<float,int>(DasvmScheme(sample), 0);
	return std::pair<float,int>(LastNodeScheme(sample), 0);
	//return NewScheme(sample);
}

float EnsembleListSvm::classifyWithCertainty(const gsl::span<const float> sample) const
{
	auto temp = root;

	float result = -50;
	std::vector<ResultAndThresholds> history;
	while (temp)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(temp->m_svm.get());
		result = res->classifyWithCertainty(sample);
		history.emplace_back(result, res->getPositiveCertainty(), res->getNegativeCertainty());

		if (result != -100)
			break;

		temp = temp->m_next;
	}

	if (result == -100)
	{
		return -100;
	}

	return result;
}

float EnsembleListSvm::classifyHyperplaneDistance(const gsl::span<const float> sample) const
{
	return classify(sample);
}

double EnsembleListSvm::classifyWithOptimalThreshold(const gsl::span<const float> sample) const
{
	return classify(sample);
}

void EnsembleListSvm::train(const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/, bool)
{
	throw std::exception("Not implemented EnsembleListSvm. Training is performed by EnsembleTreeWorkflow");
	/*std::vector<uint64_t> set;
	for (auto i = 0u; i < trainingSet.size(); ++i)
	{
	    set.emplace_back(i);
	}
	std::vector<uint64_t> vset;
	for (auto i = 0u; i < m_loadingWorkflow.getValidationSet().size(); ++i)
	{
	    vset.emplace_back(i);
	}

	auto temp = std::make_shared<phd::svm::ListNodeSvm>(nullptr);
	root = trainHelper(temp, trainingSet, set, m_loadingWorkflow.getValidationSet(), vset);*/
}
}} // namespace phd::svm
