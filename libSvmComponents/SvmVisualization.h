#pragma once

#include <map>

#include <LodePng/lodepng.h>
//#include <opencv2/core.hpp>

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



// RGB color class
struct RGBColor
{
    unsigned char r, g, b;

    RGBColor(unsigned char red, unsigned char green, unsigned char blue)
        : r(red), g(green), b(blue)
    {
    }
};

enum class MarkerType
{
    CROSS,
    TILTED_CROSS,
    CIRCLE,
    SQUARE
};

// Helper class for managing an image buffer
class ImageBuffer
{
public:
 	int rows;
    int cols;

	ImageBuffer() = default;

    ImageBuffer(int width, int height)
        : width_(width), height_(height), buffer_(width * height * 4, 255) // Initialize with fully opaque white pixels
    {
		cols = width;
		rows = height;
    }

    // Set pixel color at (x, y)
    void setPixel(int x, int y, const RGBColor& color)
    {
        if (x >= 0 && x < width_ && y >= 0 && y < height_)
        {
            int index = (y * width_ + x) * 4;
            buffer_[index + 0] = color.r; // Red component
            buffer_[index + 1] = color.g; // Green component
            buffer_[index + 2] = color.b; // Blue component
            buffer_[index + 3] = 255;     // Alpha component (fully opaque)
        }
    }


	void drawMarker(int x, int y, const RGBColor& color, MarkerType markerType, int markerSize, int markerThickness)
    {
        switch (markerType)
        {
        case MarkerType::CROSS:
            drawCross(x, y, color, markerSize, markerThickness);
            break;
        case MarkerType::TILTED_CROSS:
            drawTiltedCross(x, y, color, markerSize, markerThickness);
            break;
        case MarkerType::CIRCLE:
            drawCircle(x, y, color, markerSize, markerThickness);
            break;
        case MarkerType::SQUARE:
            drawSquare(x, y, color, markerSize, markerThickness);
            break;
        }
    }

	std::vector<unsigned char> encodeToPng()
	{
		 std::vector<unsigned char> encodedBuffer;
    	unsigned error = lodepng::encode(encodedBuffer, buffer_, width_, height_);

		if (error)
		{
			throw std::runtime_error("Error encoding PNG image with LodePNG: " + std::to_string(error));
		}

		return encodedBuffer;
	}

	int getWidth() const
    {
        return width_;
    }

    // Get the height of the image
    int getHeight() const
    {
        return height_;
    }

	//unsigned error = lodepng::encode(filename, buffer_, width_, height_);

private:
// Helper function to draw a cross marker
    void drawCross(int x, int y, const RGBColor& color, int size, int thickness)
    {
        drawLine(x - size, y, x + size, y, color, thickness);
        drawLine(x, y - size, x, y + size, color, thickness);
    }

    // Helper function to draw a tilted cross marker
    void drawTiltedCross(int x, int y, const RGBColor& color, int size, int thickness)
    {
        int halfSize = size / 2;
        drawLine(x - halfSize, y - halfSize, x + halfSize, y + halfSize, color, thickness);
        drawLine(x - halfSize, y + halfSize, x + halfSize, y - halfSize, color, thickness);
    }

    // Helper function to draw a circle marker
    void drawCircle(int x, int y, const RGBColor& color, int radius, int thickness)
    {
        int cx = x;
        int cy = y;
        int r = radius;

        for (int angle = 0; angle <= 360; ++angle)
        {
			constexpr double pi = 3.14159265358979323846;
            double radian = angle * pi  / 180.0;
            int px = static_cast<int>(std::round(cx + r * std::cos(radian)));
            int py = static_cast<int>(std::round(cy + r * std::sin(radian)));

            setPixel(px, py, color);
        }
    }

    // Helper function to draw a square marker
    void drawSquare(int x, int y, const RGBColor& color, int size, int thickness)
    {
        int halfSize = size / 2;
        drawLine(x - halfSize, y - halfSize, x + halfSize, y - halfSize, color, thickness);
        drawLine(x + halfSize, y - halfSize, x + halfSize, y + halfSize, color, thickness);
        drawLine(x + halfSize, y + halfSize, x - halfSize, y + halfSize, color, thickness);
        drawLine(x - halfSize, y + halfSize, x - halfSize, y - halfSize, color, thickness);
    }

    // Helper function to draw a line
    void drawLine(int x1, int y1, int x2, int y2, const RGBColor& color, int thickness)
    {
        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);

        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;

        int err = dx - dy;

        while (true)
        {
            setPixel(x1, y1, color);

            if (x1 == x2 && y1 == y2)
                break;

            int e2 = 2 * err;

            if (e2 > -dy)
            {
                err -= dy;
                x1 += sx;
            }

            if (e2 < dx)
            {
                err += dx;
                y1 += sy;
            }
        }
    }


    int width_;
    int height_;
    std::vector<unsigned char> buffer_;
};

using uchar = unsigned char;

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
		std::vector<std::vector<float>> lastNodeSvs, phd::svm::ISvm& no_last_node);
	
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
	void visualizeClassBoundries(phd::svm::EnsembleListSvm svm, std::vector<std::vector<RGBColor>> colors);

	void visualizeCertaintyRegions(phd::svm::ISvm& svm);
	void visualizeCertaintyRegions(phd::svm::EnsembleListSvm& svm);


	void visualizeCertaintyRegionsImprovement(phd::svm::ISvm& svm, phd::svm::ISvm& no_last_node);

	std::vector<uchar> createVisualizationCertaintyNoSV(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& testDataset);

	std::vector<uchar> createVisualizationImprovementEnsembleList(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& testDataset, std::vector<std::vector<float>> lastNodeSvs, phd::svm::ISvm& no_last_node, bool visualizeSVs);

	std::vector<uchar> differenceInClassification(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& testDataset,
		phd::svm::ISvm& no_last_node);

	void visualizeGroupsClassification(phd::svm::ISvm& svm, const dataset::Dataset<std::vector<float>, float>& testDataset);

	//default are white and black
	void visualizeTestData(const dataset::Dataset<std::vector<float>, float>& testDataset, RGBColor positiveColor = RGBColor(255, 255, 255), RGBColor negativeColor = RGBColor(0, 0, 0));
	void visualizeTraningData(const dataset::Dataset<std::vector<float>, float>& trainingDataset);
	void visualizeSupportVectors(phd::svm::ISvm& svm,
	                             const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	void visualizeImprovementSupportVectors(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset, std::vector<std::vector<float>> lastNodeSvs);
	

	std::vector<uchar> visualizeSVtoVectors(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	void visualizeSupportVectors(phd::svm::EnsembleListSvm svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset, std::vector<std::vector<RGBColor>> colors);


	std::vector<uchar> visualizeSoftBoundries(phd::svm::ISvm& svm);

	std::vector<uchar> basicVisualization(phd::svm::ISvm& svm,
	                                      const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	std::vector<uchar> visualizationWithData(phd::svm::ISvm& svm,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset,
		const dataset::Dataset<std::vector<float>, float>& testDataset);


	std::vector<uchar> createVisualizationNewTrainingSet2(phd::svm::ISvm& svm, int height, int width,
		const dataset::Dataset<std::vector<float>, float>& trainingDataset);

	RGBColor getSvColor(long long index_in_training);

	int normalizeToImageHeight(double value) const;
	int normalizeToImageWidth(double value) const;

	ImageBuffer m_image;
	const RGBColor darkGrey = RGBColor(50, 50, 50);
	const RGBColor grey = RGBColor(100, 100, 100);
	const RGBColor black = RGBColor(0, 0, 0);
	const RGBColor white = RGBColor(255, 255, 255);
	const RGBColor yellow = RGBColor(0, 255, 255);
	//const cv::Vec3b unknownColor = cv::Vec3b(227, 250, 255); //another to try 154, 170, 173
	//const cv::Vec3b unknownColor = cv::Vec3b(154, 170, 173); //another to try 
	const RGBColor unknownColor = RGBColor(105, 105, 171); //red
	
	//const cv::LineTypes lineType = cv::LineTypes::LINE_8;  //LINE_AA
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
	auto height = std::round(value * (m_image.getHeight() - 1));

	if (height < 0 || height > m_image.getHeight())
		return 0;

	return static_cast<int>(height);
}

inline int SvmVisualization::normalizeToImageWidth(double value) const
{
	auto width = std::round(value * (m_image.getWidth() - 1));

	if (width < 0 || width > m_image.getWidth())
		return 0;

	return static_cast<int>(width);
}
} // namespace svmComponents
