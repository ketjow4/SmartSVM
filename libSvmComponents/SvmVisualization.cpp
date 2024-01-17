#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/viz/types.hpp>
#include "SvmVisualization.h"

#include "SvmLib/EnsembleListSvm.h"
#include "SvmComponentsExceptions.h"
#include "SvmLib/ISvm.h"
#include "SvmLib/libSvmImplementation.h"

namespace svmComponents
{
std::vector<uchar> SvmVisualization::createVisualization(phd::svm::ISvm& svm,
                                                         int height,
                                                         int width,
                                                         const dataset::Dataset<std::vector<float>, float>& trainingDataset,
                                                         const dataset::Dataset<std::vector<float>, float>& testDataset)
{
	m_isChromosomeSetup = false;
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);


	if(trainingDataset.hasGroups())
	{
		visualizeClassBoundries(svm);
		visualizeTestData(testDataset);
		visualizeSupportVectors(svm, trainingDataset);
		visualizeGroupsClassification(svm, testDataset);
	}
	else
	{
		visualizeClassBoundries(svm);
		visualizeTestData(testDataset);
		//visualizeTraningData(trainingDataset);
		visualizeSupportVectors(svm, trainingDataset);
	}

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);
	
	return buffer;
}

std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createVisualizationNewTrainingSet(phd::svm::ISvm& svm, int height, int width,
                                                                                                            const dataset::Dataset<std::vector<float>, float>&
                                                                                                            trainingDataset)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	visualizeClassBoundries(svm);
	visualizeTestData(trainingDataset);
	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	visualizations.emplace_back(buffer, "new_training_set");

	return visualizations;
}

std::vector<uchar> SvmVisualization::createVisualizationCertainty(phd::svm::ISvm& svm,
                                                                  const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
	visualizeCertaintyRegions(svm);
	visualizeTestData(trainingDataset);
	//visualizeSupportVectors(svm, trainingDataset);

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}


std::vector<uchar> SvmVisualization::createVisualizationCertaintyNoSV(phd::svm::ISvm& svm,
	const dataset::Dataset<std::vector<float>, float>& testDataset)
{
	visualizeCertaintyRegions(svm);
	visualizeTestData(testDataset);

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}

std::vector<uchar> SvmVisualization::createVisualizationImprovementEnsembleList(phd::svm::ISvm& svm,
	const dataset::Dataset<std::vector<float>, float>& trainingDataset, cv::Mat lastNodeSvs, phd::svm::ISvm& no_last_node, bool visualizeSVs)
{
	visualizeCertaintyRegionsImprovement(svm, no_last_node);
	visualizeTestData(trainingDataset);
	if(visualizeSVs)
	{
		visualizeImprovementSupportVectors(svm, trainingDataset, lastNodeSvs);
	}

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}


std::vector<std::pair<unsigned int, float>>
getUncertainDataset(const dataset::Dataset<std::vector<float>, float>& trainingSet, phd::svm::ISvm& svm)
{
	std::vector<std::pair<unsigned int, float>> uncertainTrainingSet;


	auto samples = trainingSet.getSamples();
	//auto labels = trainingSet.getLabels();

	for (auto i = 0u; i < samples.size(); ++i)
	{
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(&svm);
		auto result = res->classifyWithCertainty(samples[i]);
		if (result == -100)   //if we are not sure we pass this example further into list
		{
			auto classification = res->classify(samples[i]);
			uncertainTrainingSet.emplace_back(i, classification);
		}
	}

	return uncertainTrainingSet;
}


std::vector<uchar> SvmVisualization::differenceInClassification(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& testDataset, phd::svm::ISvm& no_last_node)
{
	visualizeClassBoundries(svm);
	visualizeCertaintyRegions(no_last_node);
	//visualizeTestData(testDataset);
	
	//wektory niepewne mniejszego
	auto uncertainSmallerEnsemble = getUncertainDataset(testDataset, no_last_node);

	//druga kalsyfikacja wiekszym
	for(auto& v : uncertainSmallerEnsemble)
	{
		auto sampleID = v.first;
		auto sample = testDataset.getSample(sampleID);
		//auto label = testDataset.getLabel(sampleID);
		auto classification = svm.classify(sample);

		if(classification != v.second)
		{
			//wyrysowac r�znice
			//if( label == 0)
			//{ 
			//	m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = cv::Vec3b(48,81,252); //bgr red dark
			//}
			//else
			//{
				m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = cv::Vec3b(234, 234, 18); //bgr red light
			//}
		}
	}

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}



std::vector<uchar> SvmVisualization::createVisualizationNewTrainingSet2(phd::svm::ISvm& svm, int height, int width,
                                                                        const dataset::Dataset<std::vector<float>, float>& dataset)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	visualizeClassBoundries(svm);
	visualizeTestData(dataset);
	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}

std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createVisualizationNewValidationSet(int height, int width,
	const dataset::Dataset<std::vector<float>, float>& validationDataset)
{
	m_image = cv::Mat(height, width, CV_8UC3, grey);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	visualizeTestData(validationDataset);
	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	visualizations.emplace_back(buffer, "__");

	return visualizations;
}

std::vector<uchar> SvmVisualization::basicVisualization(phd::svm::ISvm& svm,
                                                        const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
	visualizeClassBoundries(svm);
	//visualizeTestData(trainingDataset);
	//visualizeTraningData(trainingDataset);
	visualizeSupportVectors(svm, trainingDataset);

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}

std::vector<uchar> SvmVisualization::visualizationWithData(phd::svm::ISvm& svm,
                                                           const dataset::Dataset<std::vector<float>, float>& /*trainingDataset*/,
                                                           const dataset::Dataset<std::vector<float>, float>& testDataset)
{
	visualizeClassBoundries(svm);
	visualizeTestData(testDataset);
	//visualizeTraningData(trainingDataset);
	//visualizeSupportVectors(svm, testDataset); //for now this is full set Tr + V here

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}

cv::Vec3b SvmVisualization::getSvColor(int64 index_in_training)
{
	if (!m_gammas.empty() && m_isChromosomeSetup)
	{
		std::vector<cv::Vec3b> colors{
			// BGR (Blue, Green Red)
			cv::Vec3b(254, 197, 28), //light blue
			cv::Vec3b(0, 0, 204), //nice red
			cv::Vec3b(51, 102, 51), //good green
			cv::Vec3b(51, 153, 204), //gold
			cv::Vec3b(204, 204, 255), //skin tone
			cv::Vec3b(190, 183, 127), //dark sky blue
			cv::Vec3b(229, 203, 212), //lavender
			//cv::Vec3b(50, 50, 50), //
			//cv::Vec3b(50, 50, 50), //
		};

		auto genes = m_chromosome.getDataset();
		std::vector<Gene> newOne;
		newOne.reserve(genes.size() + m_frozenSet.size()); // preallocate memory
		newOne.insert(newOne.end(), genes.begin(), genes.end());
		newOne.insert(newOne.end(), m_frozenSet.begin(), m_frozenSet.end());

		auto genePosition = std::find_if(newOne.begin(), newOne.end(), [index_in_training](Gene g)
		{
			return g.id == static_cast<std::uint64_t>(index_in_training);
		}) - newOne.begin();

		auto gamma = newOne[genePosition].gamma;
		auto colorIndex = std::find_if(m_gammas.begin(), m_gammas.end(), [gamma](double g)
		{
			return g == gamma;
		}) - m_gammas.begin();

		if(gamma < 0)
		{
			return cv::Vec3b(220, 56, 84); //purple
			//return cv::viz::Color::purple();
		}

		return colors[colorIndex];
	}

	if (m_isChromosomeSetup)
	{
		auto genes = m_chromosome.getDataset();
		auto genePosition = std::find_if(genes.begin(), genes.end(), [index_in_training](Gene g)
		{
			return g.id == static_cast<std::uint64_t>(index_in_training);
		}) - genes.begin();

		//fix for error in manual selection of dataset (due to notepad++ indexing from 1)
		for (int i = 1; i < 10; ++i)
		{
			if (static_cast<uint64>(genePosition) == genes.size())
			{
				genePosition = std::find_if(genes.begin(), genes.end(), [index_in_training, i](Gene g)
				{
					return g.id == static_cast<std::uint64_t>(index_in_training + i);
				}) - genes.begin();
				if (static_cast<uint64>(genePosition) != genes.size())
					break;
			}
		}

		//I add wide gammas only to training there are not a part of chromosome anymore in this case we need to return blue and go on
		if (static_cast<uint64>(genePosition) == genes.size())
		{
			return cv::Vec3b(0, 250, 255);
		}

		if (genes[genePosition].gamma == 10)
		{
			return cv::Vec3b(255, 0, 0);
		}
		else
		{
			return yellow;
		}
	}
	return yellow;
}

std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createDetailedVisualization(phd::svm::ISvm& svm,
                                                                                                      int height,
                                                                                                      int width,
                                                                                                      const dataset::Dataset<std::vector<float>, float>&
                                                                                                      trainingDataset,
                                                                                                      const dataset::Dataset<std::vector<float>, float>&
                                                                                                      validationDataset)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	//visualizations.emplace_back(basicVisualization(svm, trainingDataset, validationDataset), "basic");
	visualizations.emplace_back(visualizeSoftBoundries(svm), "softBoundaries");
	visualizations.emplace_back(visualizationWithData(svm, trainingDataset, validationDataset), "withValidation");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, trainingDataset), "NoSV_TrainingSet");

	visualizations.emplace_back(createVisualizationCertainty(svm, trainingDataset), "CertaintyRegion");
	visualizations.emplace_back(createVisualizationCertaintyNoSV(svm, validationDataset), "CertaintyRegionValidationSet");
	

	//markerThickness = 2;
	//markerCrossSize = 9;  //16
	//markerTiltedCrossSize = 7;  //14
	//visualizations.emplace_back(visualizationWithData(svm, trainingDataset, validationDataset), "SmallerSVs");

	
	//m_image = cv::Mat::zeros(height, width, CV_8UC3);
	//visualizations.emplace_back(visualizeSVtoVectors(svm, trainingDataset), "mapSV_to_vectors");

	return visualizations;
}

std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createDetailedVisualization(phd::svm::ISvm& svm, int height, int width,
	const dataset::Dataset<std::vector<float>, float>& trainingDataset, 
	const dataset::Dataset<std::vector<float>, float>& validationDataset,
	const dataset::Dataset<std::vector<float>, float>& testDataset)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	//visualizations.emplace_back(basicVisualization(svm, trainingDataset, testDataset), "basic");
	//m_image = cv::Mat::zeros(height, width, CV_8UC3);
	visualizations.emplace_back(visualizeSoftBoundries(svm), "softBoundaries");
	//m_image = cv::Mat::zeros(height, width, CV_8UC3);
	visualizations.emplace_back(visualizationWithData(svm, trainingDataset, testDataset), "withTest");
	visualizations.emplace_back(visualizationWithData(svm, trainingDataset, validationDataset), "withValidation");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, trainingDataset), "NoSV_TrainingSet");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, validationDataset), "NoSV_Validation");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, testDataset), "NoSV_Test");

	//visualizations.emplace_back(createVisualizationCertainty(svm, trainingDataset), "CertaintyRegion");
	//visualizations.emplace_back(createVisualizationCertaintyNoSV(svm, validationDataset), "CertaintyRegionValidationSet");
	//visualizations.emplace_back(createVisualizationCertaintyNoSV(svm, testDataset), "CertaintyRegionTestSet");


	//markerThickness = 2;
	//markerCrossSize = 9;  //16
	//markerTiltedCrossSize = 7;  //14
	//visualizations.emplace_back(visualizationWithData(svm, trainingDataset, testDataset), "SmallerSVs");


	//m_image = cv::Mat::zeros(height, width, CV_8UC3);
	//visualizations.emplace_back(visualizeSVtoVectors(svm, trainingDataset), "mapSV_to_vectors");

	return visualizations;
}

std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createEnsembleVisualization(phd::svm::ISvm& svm, int height, int width,
                                                                                                      const dataset::Dataset<std::vector<float>, float>&
                                                                                                      trainingDataset,
                                                                                                      const dataset::Dataset<std::vector<float>, float>&
                                                                                                      validationDataset,
                                                                                                      const dataset::Dataset<std::vector<float>, float>&
                                                                                                      testDataset)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;


	//m_image = cv::Mat::zeros(height, width, CV_8UC3);
	visualizations.emplace_back(visualizationWithData(svm, trainingDataset, validationDataset), "withValidationData");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, trainingDataset), "NoSV_TrainingSet");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, validationDataset), "NoSV_ValidationSet");
	visualizations.emplace_back(createVisualizationNewTrainingSet2(svm, height, width, testDataset), "NoSV_Test");

	visualizations.emplace_back(createVisualizationCertainty(svm, trainingDataset), "CertaintyRegion");
	visualizations.emplace_back(createVisualizationCertaintyNoSV(svm, validationDataset), "CertaintyRegionValidationSet");
	visualizations.emplace_back(createVisualizationCertaintyNoSV(svm, testDataset), "CertaintyRegionTestSet");

	return visualizations;
}


std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createEnsembleVisualizationCertaintyMap(phd::svm::EnsembleListSvm& svm, int height, int width)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;


	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	visualizeCertaintyRegions(svm);
	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);
	visualizations.emplace_back(buffer, "_certaintyMap");

	return visualizations;
}


std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createEnsembleVisualizationPerNode(phd::svm::EnsembleListSvm svm, int height,
	int width, const dataset::Dataset<std::vector<float>, float>& trainingDataset, const dataset::Dataset<std::vector<float>, float>& validationDataset,
	const dataset::Dataset<std::vector<float>, float>& testDataset)
{


	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	visualizations.emplace_back(createEnsembleVisualizationPerNodeInternal(svm, 500,500, trainingDataset, validationDataset, testDataset), "_NewScheme_");
	
	return visualizations;
	
}

cv::Vec3b rgbToCvColor(uchar r, uchar g, uchar b)
{
	return cv::Vec3b(b - 30, g - 30, r - 30);
}

std::vector<uchar> SvmVisualization::createEnsembleVisualizationPerNodeInternal(phd::svm::EnsembleListSvm svm, int , int ,
	const dataset::Dataset<std::vector<float>, float>& /*trainingDataset*/, const dataset::Dataset<std::vector<float>, float>& /*validationDataset*/,
	const dataset::Dataset<std::vector<float>, float>& test)
{
	//get constant colors for each node
	std::vector<std::vector<cv::Vec3b>> colors = 
	{

	/*	{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},
		{grey, yellow, darkGrey},*/

		{rgbToCvColor(212, 212, 212), rgbToCvColor(128, 128, 128), rgbToCvColor(42, 42, 42)},
		{rgbToCvColor(231, 238, 187), rgbToCvColor(184, 205, 50),  rgbToCvColor(134, 215, 60)},
		{rgbToCvColor(255, 228, 170), rgbToCvColor(255, 228, 170), rgbToCvColor(245, 208, 150)},
		{rgbToCvColor(196, 220, 229), rgbToCvColor(78, 149, 177),  rgbToCvColor(70, 129, 197)},
		{rgbToCvColor(254, 171, 173), rgbToCvColor(253, 2, 9),     rgbToCvColor(253, 60, 80)},
		{rgbToCvColor(171, 206, 254), rgbToCvColor(3, 109, 252),   rgbToCvColor(1, 36, 84)},
		{rgbToCvColor(214, 185, 240), rgbToCvColor(214, 185, 240), rgbToCvColor(184, 145, 240)}
	};

	
	//visualize per node data with proper color
	//visualizeClassBoundries(svm, colors);
	visualizeClassBoundries(svm);
	//visualizeSupportVectors(svm, trainingDataset, colors);
	//visualizeTestData(validationDataset);
	visualizeTestData(test);

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}

std::vector<std::pair<std::vector<uchar>, std::string>> SvmVisualization::createEnsembleImprobementVisualization(
	phd::svm::ISvm& svm, int height, int width,
	const dataset::Dataset<std::vector<float>, float>& trainingDataset,
	const dataset::Dataset<std::vector<float>, float>& validationDataset,
	const dataset::Dataset<std::vector<float>, float>& testDataset, 
	cv::Mat lastNodeSvs,
	phd::svm::ISvm& no_last_node)
{
	if (!svm.isTrained())
	{
		throw UntrainedSvmClassifierException();
	}
	m_image = cv::Mat::zeros(height, width, CV_8UC3);

	std::vector<std::pair<std::vector<uchar>, std::string>> visualizations;

	visualizations.emplace_back(createVisualizationImprovementEnsembleList(svm, trainingDataset, lastNodeSvs, no_last_node, false), "1_Improvement");
	visualizations.emplace_back(createVisualizationImprovementEnsembleList(svm, trainingDataset, lastNodeSvs, no_last_node, false), "2_Improvement_NoSV");
	visualizations.emplace_back(createVisualizationImprovementEnsembleList(svm, validationDataset, lastNodeSvs, no_last_node, false), "3_Improvement_NoSV_ValidationSet");
	visualizations.emplace_back(createVisualizationImprovementEnsembleList(svm, testDataset, lastNodeSvs, no_last_node, false), "4_Improvement_NoSV_TestSet");

	//visualizations.emplace_back(differenceInClassification(svm,testDataset,no_last_node), "4_difference_in_uncertain");
	
	return visualizations;
	
}

void SvmVisualization::visualizeCertaintyRegions(phd::svm::ISvm& svm)
{
#pragma omp parallel for
	for (int i = 0; i < m_image.rows * m_image.cols; i++)
	{
		int w = i / m_image.cols;
		int h = i % m_image.cols;

		std::vector<float> sample = { static_cast<float>(h) / static_cast<float>(m_image.cols), static_cast<float>(w) / static_cast<float>(m_image.rows) };

	
		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(&svm);
		auto response = res->classifyWithCertainty(sample);

		if (response == 1)
		{
			m_image.at<cv::Vec3b>(w, h) = grey;
		}
		else if (response == 0)
		{
			m_image.at<cv::Vec3b>(w, h) = darkGrey;
		}
		else if (response == -100)
		{
			m_image.at<cv::Vec3b>(w, h) = unknownColor;
		}
	}
}

void SvmVisualization::visualizeCertaintyRegions(phd::svm::EnsembleListSvm& svm)
{
#pragma omp parallel for
	for (int i = 0; i < m_image.rows * m_image.cols; i++)
	{
		int w = i / m_image.cols;
		int h = i % m_image.cols;


		std::vector<float> sample = { static_cast<float>(h) / static_cast<float>(m_image.cols), static_cast<float>(w) / static_cast<float>(m_image.rows) };

		auto response = svm.classifyWithCertainty(sample);

		if (response == 1)
		{
			m_image.at<cv::Vec3b>(w, h) = grey;
		}
		else if (response == 0)
		{
			m_image.at<cv::Vec3b>(w, h) = darkGrey;
		}
		else if (response == -100)
		{
			m_image.at<cv::Vec3b>(w, h) = unknownColor;
		}
	}
}


void SvmVisualization::visualizeCertaintyRegionsImprovement(phd::svm::ISvm& svm, phd::svm::ISvm& no_last_node)
{
	std::vector<std::pair<int, int>> uncertain;
#pragma omp parallel for
	for (int i = 0; i < m_image.rows * m_image.cols; i++)
	{
		int w = i / m_image.cols;
		int h = i % m_image.cols;

		std::vector<float> sample = { static_cast<float>(h) / static_cast<float>(m_image.cols), static_cast<float>(w) / static_cast<float>(m_image.rows) };


		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(&no_last_node);
		auto response = res->classifyWithCertainty(sample);

		if (response == 1)
		{
			m_image.at<cv::Vec3b>(w, h) = grey;
		}
		else if (response == 0)
		{
			m_image.at<cv::Vec3b>(w, h) = darkGrey;
		}
		else if (response == -100)
		{
			m_image.at<cv::Vec3b>(w, h) = unknownColor;
#pragma omp critical
			uncertain.emplace_back(w, h);
		}
	}

#pragma omp parallel for
	for (int i = 0; i < uncertain.size(); i++)
	{
		auto coordinates = uncertain[i];
		std::vector<float> sample = { static_cast<float>(coordinates.second) / static_cast<float>(m_image.cols), static_cast<float>(coordinates.first) / static_cast<float>(m_image.rows) };

		auto res = reinterpret_cast<phd::svm::libSvmImplementation*>(&svm);
		auto response = res->classifyWithCertainty(sample);

		if (response == 1)
		{
			m_image.at<cv::Vec3b>(coordinates.first, coordinates.second) = cv::Vec3b(114, 180, 213);
		}
		else if (response == 0)
		{
			m_image.at<cv::Vec3b>(coordinates.first, coordinates.second) = cv::Vec3b(39, 152, 180); //bgr
		}
		else if (response == -100)
		{
			m_image.at<cv::Vec3b>(coordinates.first, coordinates.second) = unknownColor;

		}
	}
	
}


void SvmVisualization::visualizeClassBoundries(phd::svm::EnsembleListSvm svm, std::vector<std::vector<cv::Vec3b>> colors)
{
#pragma omp parallel for
	for (int i = 0; i < m_image.rows * m_image.cols; i++)
	{
		int w = i / m_image.cols;
		int h = i % m_image.cols;

		std::vector<float> sample = { static_cast<float>(h) / static_cast<float>(m_image.cols), static_cast<float>(w) / static_cast<float>(m_image.rows) };

		double response = -1;
		auto result = svm.classifyWithNode(sample);
		response = result.first;

		auto color = colors[result.second % colors.size()];
		
		if (response == 1)
		{
			m_image.at<cv::Vec3b>(w, h) = color[0];
		}
		else
		{
			m_image.at<cv::Vec3b>(w, h) = color[2];
		}
	}
}



void SvmVisualization::visualizeClassBoundries(phd::svm::ISvm& svm)
{
#pragma omp parallel for
	for (int i = 0; i < m_image.rows * m_image.cols; i++)
	{
		int w = i / m_image.cols;
		int h = i % m_image.cols;

		std::vector<float> sample = {static_cast<float>(h) / static_cast<float>(m_image.cols), static_cast<float>(w) / static_cast<float>(m_image.rows)};

		double response = -1;
		if(svm.canClassifyWithOptimalThreshold())
			response = svm.classifyWithOptimalThreshold(sample);
		else
			response = svm.classify(sample);
		
		if (response == 1)
		{
			m_image.at<cv::Vec3b>(w, h) = grey;
		}
		else
		{
			m_image.at<cv::Vec3b>(w, h) = darkGrey;
		}
	}
}


void SvmVisualization::visualizeGroupsClassification(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& testDataset)
{
	auto samples = testDataset.getSamples();
	const auto targets = testDataset.getLabels();
	/*std::vector<double> results;
	results.reserve(targets.size());*/

	auto answers = svm.classifyGroups(testDataset);
	auto groups = testDataset.getGroups();

	//if group helped
	cv::Vec3b positiveColor = cv::Vec3b(118, 221, 119);  // (119, 221, 118)
	//if groups worsen the result
	cv::Vec3b negativeColor = cv::Vec3b(98, 105, 255);  //RGB: (255, 105, 98)

	for (auto i = 0U; i < testDataset.size(); i++)
	{
		auto sample = samples[i];
		const auto result = svm.classifyWithOptimalThreshold(sample);
		const auto groupAnswer = answers[static_cast<int>(groups[i])];

		if (groupAnswer != result)
		{
			if (static_cast<int>(targets[i]) == groupAnswer)
			{
				m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = positiveColor;
			}
			else
			{
				m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = negativeColor;
			}
		}
		if(groupAnswer == result && result != targets[i])
		{
			m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = cv::Vec3b(190,119,0);
		}

	}
}

void SvmVisualization::visualizeTestData(const dataset::Dataset<std::vector<float>, float>& testDataset, cv::Vec3b positiveColor, cv::Vec3b negativeColor)
{
	auto samples = testDataset.getSamples();
	const auto targets = testDataset.getLabels();
	for (auto i = 0U; i < testDataset.size(); i++)
	{
		auto sample = samples[i];
		auto target = targets[i];
		if (target == 1)
		{
			m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = positiveColor;
			/*m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]) + 1, normalizeToImageWidth(sample[0] ) + 1) = white;
			m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0] ) +1) = white;
			m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1] ) +1, normalizeToImageWidth(sample[0])) = white;*/
		}
		else
		{
			m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0])) = negativeColor;
			//m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]) + 1, normalizeToImageWidth(sample[0]) + 1) = black;
			//m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]), normalizeToImageWidth(sample[0]) + 1) = black;
			//m_image.at<cv::Vec3b>(normalizeToImageHeight(sample[1]) + 1, normalizeToImageWidth(sample[0])) = black;
		}
	}
}

void SvmVisualization::visualizeTraningData(const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
	auto samples = trainingDataset.getSamples();
	auto targets = trainingDataset.getLabels();
	for (auto i = 0; i < samples.size(); i++)
	{
		auto sample = samples[i];
		auto target = targets[i];
		if (target == 1)
		{
			cv::drawMarker(m_image,
			               cv::Point(normalizeToImageWidth(sample[0]), normalizeToImageHeight(sample[1])),
			               black,
			               cv::MARKER_CROSS,
			               markerCrossSize,
			               markerThickness,
			               lineType);
		}
		else
		{
			cv::drawMarker(m_image,
			               cv::Point(normalizeToImageWidth(sample[0]), normalizeToImageHeight(sample[1])),
			               white,
			               cv::MARKER_TILTED_CROSS,
			               markerTiltedCrossSize,
			               markerThickness,
			               lineType);
		}
	}
}

void SvmVisualization::visualizeSupportVectors(phd::svm::ISvm& svm,
                                               const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
	// const auto samples = trainingDataset.getSamples();
	// const auto targets = trainingDataset.getLabels();
	// auto supportVectors = svm.getSupportVectors();

	// //auto libSvm = reinterpret_cast<phd::svm::libSvmImplementation*>(&svm);
	// for (int i = 0; i < supportVectors.rows; i++)
	// {
	// 	const float* sv = supportVectors.ptr<float>(i);
	// 	constexpr auto epsilon = 1e-3; //0.001;
	// 	auto positionSv = std::find_if(samples.begin(), samples.end(), [&sv,&epsilon](auto& sample)
	// 	{
	// 		return abs(sample[1] - sv[1]) < epsilon && abs(sample[0] - sv[0]) < epsilon;
	// 	}) - samples.begin();

	// 	if (targets[positionSv] == 1)
	// 	{
	// 		cv::drawMarker(m_image,
	// 		               cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 		               getSvColor(positionSv),
	// 		               cv::MARKER_CROSS,
	// 		               markerCrossSize,
	// 		               markerThickness,
	// 		               lineType);
	// 		if (!m_scores.empty())
	// 			cv::putText(m_image, std::to_string(m_scores[i]), cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])), 0, 0.5,
	// 			            cv::viz::Color::red(), 1);
	// 	/*	if (!m_gammas.empty() && m_isChromosomeSetup)
	// 		{
	// 			cv::putText(m_image, std::to_string(static_cast<int>(m_chromosome.getDataset()[i].gamma)), cv::Point(normalizeToImageWidth(sv[0])+5, normalizeToImageHeight(sv[1]) + 5), 0, 0.5,
	// 				cv::viz::Color::red(), 1);
	// 		}*/
	// 	}
	// 	else
	// 	{
	// 		cv::drawMarker(m_image,
	// 		               cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 		               getSvColor(positionSv),
	// 		               cv::MARKER_TILTED_CROSS,
	// 		               markerTiltedCrossSize,
	// 		               markerThickness,
	// 		               lineType);
	// 		if (!m_scores.empty())
	// 			cv::putText(m_image, std::to_string(m_scores[i]), cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])), 0, 0.5,
	// 			            cv::viz::Color::red(), 1);
	// 		/*if (!m_gammas.empty() && m_isChromosomeSetup)
	// 		{
	// 			cv::putText(m_image, std::to_string(static_cast<int>(m_chromosome.getDataset()[i].gamma)), cv::Point(normalizeToImageWidth(sv[0])+5, normalizeToImageHeight(sv[1]) + 5), 0, 0.5,
	// 				cv::viz::Color::red(), 1);
	// 		}*/
	// 	}
	// }
}


void SvmVisualization::visualizeSupportVectors(phd::svm::EnsembleListSvm svm,
	const dataset::Dataset<std::vector<float>, float>& trainingDataset, std::vector<std::vector<cv::Vec3b>> colors)
{
	// const auto samples = trainingDataset.getSamples();
	// const auto targets = trainingDataset.getLabels();

	// auto temp = svm.root;
	// auto colorID = 0;
	// while(temp)
	// {
	// 	auto supportVectors = temp->m_svm->getSupportVectors();
		
	// 	for (int i = 0; i < supportVectors.rows; i++)
	// 	{
	// 		const float* sv = supportVectors.ptr<float>(i);
	// 		constexpr auto epsilon = 1e-3; //0.001;
	// 		auto positionSv = std::find_if(samples.begin(), samples.end(), [&sv, &epsilon](auto& sample)
	// 			{
	// 				return abs(sample[1] - sv[1]) < epsilon && abs(sample[0] - sv[0]) < epsilon;
	// 			}) - samples.begin();

	// 			if (targets[positionSv] == 1)
	// 			{
	// 				cv::drawMarker(m_image,
	// 					cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 					colors[colorID][1],
	// 					cv::MARKER_CROSS,
	// 					markerCrossSize,
	// 					markerThickness,
	// 					lineType);
	// 			}
	// 			else
	// 			{
	// 				cv::drawMarker(m_image,
	// 					cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 					colors[colorID][1],
	// 					cv::MARKER_TILTED_CROSS,
	// 					markerTiltedCrossSize,
	// 					markerThickness,
	// 					lineType);
	// 			}
	// 	}
	// 	colorID++;
	// 	colorID = colorID % colors.size();
	// 	temp = temp->m_next;
	// }
}

void SvmVisualization::visualizeImprovementSupportVectors(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& trainingDataset, cv::Mat lastNodeSvs)
{
	// const auto samples = trainingDataset.getSamples();
	// const auto targets = trainingDataset.getLabels();
	// auto supportVectors = svm.getSupportVectors();

	// //auto libSvm = reinterpret_cast<phd::svm::libSvmImplementation*>(&svm);
	// for (int i = 0; i < supportVectors.rows; i++)
	// {
	// 	const float* sv = supportVectors.ptr<float>(i);
	// 	constexpr auto epsilon = 1e-3; //0.001;
	// 	auto positionSv = std::find_if(samples.begin(), samples.end(), [&sv, &epsilon](auto& sample)
	// 		{
	// 			return abs(sample[1] - sv[1]) < epsilon && abs(sample[0] - sv[0]) < epsilon;
	// 		}) - samples.begin();

	// 		if (targets[positionSv] == 1)
	// 		{
	// 			cv::drawMarker(m_image,
	// 				cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 				getSvColor(positionSv),
	// 				cv::MARKER_CROSS,
	// 				markerCrossSize,
	// 				markerThickness,
	// 				lineType);
	// 		}
	// 		else
	// 		{
	// 			cv::drawMarker(m_image,
	// 				cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 				getSvColor(positionSv),
	// 				cv::MARKER_TILTED_CROSS,
	// 				markerTiltedCrossSize,
	// 				markerThickness,
	// 				lineType);
	// 		}
	// }

	// supportVectors = lastNodeSvs;

	// for (int i = 0; i < supportVectors.rows; i++)
	// {
	// 	const float* sv = supportVectors.ptr<float>(i);
	// 	constexpr auto epsilon = 1e-3; //0.001;
	// 	auto positionSv = std::find_if(samples.begin(), samples.end(), [&sv, &epsilon](auto& sample)
	// 		{
	// 			return abs(sample[1] - sv[1]) < epsilon && abs(sample[0] - sv[0]) < epsilon;
	// 		}) - samples.begin();

	// 		if (targets[positionSv] == 1)
	// 		{
	// 			cv::drawMarker(m_image,
	// 				cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 				cv::Vec3b(254, 197, 28),
	// 				cv::MARKER_CROSS,
	// 				markerCrossSize,
	// 				markerThickness,
	// 				lineType);
	// 		}
	// 		else
	// 		{
	// 			cv::drawMarker(m_image,
	// 				cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 				cv::Vec3b(254, 197, 28),
	// 				cv::MARKER_TILTED_CROSS,
	// 				markerTiltedCrossSize,
	// 				markerThickness,
	// 				lineType);
	// 		}
	// }
}

std::vector<uchar> SvmVisualization::visualizeSVtoVectors(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& trainingDataset)
{
	visualizeClassBoundries(svm);

	// std::vector<cv::Vec3b> colors{
	// 	cv::viz::Color::blue(), cv::viz::Color::red(), cv::viz::Color::green(),
	// 	cv::viz::Color::cherry(), cv::viz::Color::pink(), cv::viz::Color::purple(),
	// 	cv::viz::Color::yellow(), cv::viz::Color::lime(), cv::viz::Color::magenta(), cv::viz::Color::apricot(), cv::viz::Color::silver(),
	// 	cv::viz::Color::raspberry()
	// };

	// const auto samples = trainingDataset.getSamples();
	// const auto targets = trainingDataset.getLabels();
	// auto supportVectors = svm.getSupportVectors();


	// for (auto value : m_SvToVec)
	// {
	// 	auto sample = samples[value.second];
	// 	{
	// 		cv::drawMarker(m_image,
	// 			cv::Point(normalizeToImageWidth(sample[0]), normalizeToImageHeight(sample[1])),
	// 			colors[value.first % colors.size()],
	// 			cv::MARKER_TILTED_CROSS,
	// 			3,
	// 			markerThickness,
	// 			lineType);
	// 	}
	// }


	// for (int i = 0; i < supportVectors.rows; i++)
	// {
	// 	const float* sv = supportVectors.ptr<float>(i);
	// 	constexpr auto epsilon = 0.001;
	// 	auto positionSv = std::find_if(samples.begin(), samples.end(), [&sv, &epsilon](auto& sample)
	// 	{
	// 		return abs(sample[1] - sv[1]) < epsilon && abs(sample[0] - sv[0]) < epsilon;
	// 	}) - samples.begin();

	// 	if (targets[positionSv] == 1)
	// 	{
	// 		cv::drawMarker(m_image,
	// 		               cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 		               colors[i % colors.size()],
	// 		               cv::MARKER_CROSS,
	// 		               markerCrossSize,
	// 		               markerThickness,
	// 		               lineType);
	// 	}
	// 	else
	// 	{
	// 		cv::drawMarker(m_image,
	// 		               cv::Point(normalizeToImageWidth(sv[0]), normalizeToImageHeight(sv[1])),
	// 		               colors[i % colors.size()],
	// 		               cv::MARKER_TILTED_CROSS,
	// 		               markerTiltedCrossSize,
	// 		               markerThickness,
	// 		               lineType);
	// 	}
	// }

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}

std::vector<uchar> SvmVisualization::visualizeSoftBoundries(phd::svm::ISvm& svm)
{
	std::vector<float> responses;
	responses.resize(m_image.rows * m_image.cols);

#pragma omp parallel for
	for (int i = 0; i < m_image.rows * m_image.cols; i++)
	{
		int w = i / m_image.cols;
		int h = i % m_image.cols;

		std::vector<float> sample = {static_cast<float>(h) / m_image.cols, static_cast<float>(w) / m_image.rows};

		responses[i] = (svm.classifyHyperplaneDistance(sample));
	}

	double min = *std::min_element(responses.begin(), responses.end()); //black
	double max = *std::max_element(responses.begin(), responses.end()); //white

	bool isLinear = true;
	for(auto f = 0u; f < m_gammas.size(); f++)
	{
		if (m_gammas[f] != -1)
		{ 
			isLinear = false;
			break;
		}
	}

	/*if(isLinear)
	{
		std::cout << "Linear     Min response: " << min << "     Max response:  " << max << "\n";
	}
	else
	{
		std::cout << "RBF + L     Min response: " << min << "     Max response:  " << max << "\n";
	}*/
	
	
	//double min = -1.0; //black
	//double max = 1.0; //white

	for (int i = 0; i < m_image.rows; i++)
		for (int j = 0; j < m_image.cols; j++)
		{
			std::vector<float> sample = {static_cast<float>(j) / m_image.cols, static_cast<float>(i) / m_image.rows};

			const auto input = responses[i * m_image.rows + j];

			uchar colorValue = 0;

			if (input > max)
			{
				colorValue = 255;
			}
			else if (input < min)
			{
				colorValue = 0;
			}
			else
			{
				colorValue = static_cast<uchar>(((input - min) / (max - min)) * (255 - 0) + 0); //255 - white 0 - black
			}

			m_image.at<cv::Vec3b>(i, j) = cv::Vec3b(colorValue, colorValue, colorValue);
		}

	std::vector<uchar> buffer;
	cv::imencode(".png", m_image, buffer);

	return buffer;
}
} // namespace svmComponents
