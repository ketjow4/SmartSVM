#pragma once

#include <map>
#include <opencv2/core.hpp>
#include "libDataset/Dataset.h"
#include "SvmLib/ISvm.h"
#include "libSvmComponents/SvmCustomKernelChromosome.h"

namespace phd {namespace svm {
class EnsembleListSvm;
}}

namespace svmComponents
{
enum class imageFormat
{
	png
};

class SvmVisualization
{
public:
	std::vector<uchar> createVisualization(phd::svm::ISvm& svm,
	                                       int height,
	                                       int width,
	                                       const dataset::Dataset<std::vector<float>, float>& trainingDataset,
	                                       const dataset::Dataset<std::vector<float>, float>& testDataset);

	std::vector<std::pair<std::vector<uchar>, std::string>> createDetailedVisualization(phd::svm::ISvm& svm,
	                                                                                    int height,
	                                                                                    int width,
	                                                                                    const dataset::Dataset<std::vector<float>, float>& trainingDataset,
	                                                                                    const dataset::Dataset<std::vector<float>, float>& validationDataset);

	std::vector<std::pair<std::vector<uchar>, std::string>> createDetailedVisualization(phd::svm::ISvm& svm,
		int height,
		int width,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const dataset::Dataset<std::vector<float>, float>& validationDataset ,
		const dataset::Dataset<std::vector<float>, float>& testDataset);


	std::vector<std::pair<std::vector<uchar>, std::string>> createEnsembleVisualization(phd::svm::ISvm& svm,
		int height,
		int width,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const dataset::Dataset<std::vector<float>, float>& validationDataset,
		const dataset::Dataset<std::vector<float>, float>& testDataset);

	std::vector<std::pair<std::vector<uchar>, std::string>> createEnsembleVisualizationCertaintyMap(phd::svm::EnsembleListSvm& svm, int height, int width);

	std::vector<std::pair<std::vector<uchar>, std::string>> createEnsembleVisualizationPerNode(phd::svm::EnsembleListSvm svm,
	                                                                                           int height,
	                                                                                           int width,
	                                                                                           const dataset::Dataset<std::vector<float>, float>& trainingDataset,
	                                                                                           const dataset::Dataset<std::vector<float>, float>& validationDataset,
	                                                                                           const dataset::Dataset<std::vector<float>, float>& testDataset);

	std::vector<uchar> createEnsembleVisualizationPerNodeInternal(phd::svm::EnsembleListSvm svm,
		int height,
		int width,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const dataset::Dataset<std::vector<float>, float>& validationDataset,
		const dataset::Dataset<std::vector<float>, float>& testDataset);


	std::vector<std::pair<std::vector<uchar>, std::string>> createEnsembleImprobementVisualization(phd::svm::ISvm& svm,
		int height,
		int width,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const dataset::Dataset<std::vector<float>, float>& validationDataset,
		const dataset::Dataset<std::vector<float>, float>& testDataset,
		cv::Mat lastNodeSvs, phd::svm::ISvm& no_last_node);
	
	std::vector<std::pair<std::vector<uchar>, std::string>> createVisualizationNewTrainingSet(phd::svm::ISvm& svm,
	                                                                                          int height,
	                                                                                          int width,
	                                                                                          const dataset::Dataset<std::vector<float>, float>&
	                                                                                          trainingDataset);

	std::vector<uchar> createVisualizationCertainty(phd::svm::ISvm& svm,
													const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	std::vector<std::pair<std::vector<uchar>, std::string>> createVisualizationNewValidationSet(int height, int width,
	                                                                                            const dataset::Dataset<std::vector<float>, float>&
	                                                                                            validationDataset);

	void setGammasValues(std::vector<double> gammas)
	{
		m_gammas = gammas;
	}

	void setMap(std::multimap<int, int> sv_to_vec)
	{
		m_SvToVec = sv_to_vec;
	}

	void setScores(std::vector<double> vec)
	{
		m_scores = vec;
	}

	void setGene(const SvmCustomKernelChromosome& chromosome)
	{
		m_chromosome = chromosome;
		m_isChromosomeSetup = true;
	}

	void setFrozenSet(std::vector<svmComponents::Gene> frozenSet)
	{
		m_frozenSet = frozenSet;
	}

private:	
	void visualizeClassBoundries(phd::svm::ISvm& svm);
	void visualizeClassBoundries(phd::svm::EnsembleListSvm svm, std::vector<std::vector<cv::Vec3b>> colors);

	void visualizeCertaintyRegions(phd::svm::ISvm& svm);
	void visualizeCertaintyRegions(phd::svm::EnsembleListSvm& svm);


	void visualizeCertaintyRegionsImprovement(phd::svm::ISvm& svm, phd::svm::ISvm& no_last_node);

	std::vector<uchar> createVisualizationCertaintyNoSV(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& testDataset);

	std::vector<uchar> createVisualizationImprovementEnsembleList(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& testDataset, cv::Mat lastNodeSvs, phd::svm::ISvm& no_last_node, bool visualizeSVs);

	std::vector<uchar> differenceInClassification(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& testDataset,
		phd::svm::ISvm& no_last_node);

	void visualizeGroupsClassification(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& testDataset);

	//default are white and black
	void visualizeTestData(const dataset::Dataset<std::vector<float>, float>& testDataset, cv::Vec3b positiveColor = cv::Vec3b(255, 255, 255), cv::Vec3b negativeColor = cv::Vec3b(0, 0, 0));
	void visualizeTraningData(const dataset::Dataset<std::vector<float>, float>& trainingDataset);
	void visualizeSupportVectors(phd::svm::ISvm& svm,
	                             const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	void visualizeImprovementSupportVectors(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset, cv::Mat lastNodeSvs);
	

	std::vector<uchar> visualizeSVtoVectors(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	void visualizeSupportVectors(phd::svm::EnsembleListSvm svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset, std::vector<std::vector<cv::Vec3b>> colors);


	std::vector<uchar> visualizeSoftBoundries(phd::svm::ISvm& svm);

	std::vector<uchar> basicVisualization(phd::svm::ISvm& svm,
	                                      const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	std::vector<uchar> visualizationWithData(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const dataset::Dataset<std::vector<float>, float>& testDataset);


	std::vector<uchar> createVisualizationNewTrainingSet2(phd::svm::ISvm& svm, int height, int width,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	cv::Vec3b getSvColor(int64 index_in_training);

	int normalizeToImageHeight(double value) const;
	int normalizeToImageWidth(double value) const;

	cv::Mat m_image;
	const cv::Vec3b darkGrey = cv::Vec3b(50, 50, 50);
	const cv::Vec3b grey = cv::Vec3b(100, 100, 100);
	const cv::Vec3b black = cv::Vec3b(0, 0, 0);
	const cv::Vec3b white = cv::Vec3b(255, 255, 255);
	const cv::Vec3b yellow = cv::Vec3b(0, 255, 255);
	//const cv::Vec3b unknownColor = cv::Vec3b(227, 250, 255); //another to try 154, 170, 173
	//const cv::Vec3b unknownColor = cv::Vec3b(154, 170, 173); //another to try 
	const cv::Vec3b unknownColor = cv::Vec3b(105, 105, 171); //red
	
	const cv::LineTypes lineType = cv::LineTypes::LINE_8;  //LINE_AA
	int markerThickness = 2;
	int markerCrossSize = 13;  //16
	int markerTiltedCrossSize = 11;  //14

	std::multimap<int, int> m_SvToVec;
	std::vector<double> m_scores;
	SvmCustomKernelChromosome m_chromosome;
	bool m_isChromosomeSetup = false;

	std::vector<double> m_gammas;
	std::vector<svmComponents::Gene> m_frozenSet;

};



inline int SvmVisualization::normalizeToImageHeight(double value) const
{
	auto height = std::round(value * (m_image.rows - 1));

	if (height < 0 || height > m_image.rows)
		return 0;

	return static_cast<int>(height);
}

inline int SvmVisualization::normalizeToImageWidth(double value) const
{
	auto width = std::round(value * (m_image.cols - 1));

	if (width < 0 || width > m_image.cols)
		return 0;

	return static_cast<int>(width);
}
} // namespace svmComponents
