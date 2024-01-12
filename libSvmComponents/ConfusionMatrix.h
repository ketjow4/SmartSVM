

#pragma once

#include <array>
#include <vector>
#include <sstream>
#include "SvmLib/ISvm.h"

namespace dataset
{
	template <typename Sample, typename Label, typename desc>
	class Dataset;
}

namespace svmComponents
{
class BaseSvmChromosome;

class ConfusionMatrix
{
public:
    ConfusionMatrix(const BaseSvmChromosome& individual,
                    const dataset::Dataset<std::vector<float>, float, void>& testSamples);

    ConfusionMatrix(phd::svm::ISvm& svm,
                    const dataset::Dataset<std::vector<float>, float, void>& testSamples);

    ConfusionMatrix(const BaseSvmChromosome& individual,
					const dataset::Dataset<std::vector<float>, float, void>& testSamples, bool parrarelCacl);

    ConfusionMatrix(uint32_t truePositive,
                    uint32_t trueNegative,
                    uint32_t falsePositive,
                    uint32_t falseNegative);


    ConfusionMatrix(std::array<std::array<uint32_t, 2>, 2> array)
        : m_matrix(array)
    {
    }

    double accuracy() const
    {
        return static_cast<double>(truePositive() + trueNegative()) / static_cast<double>(trueNegative() + truePositive() + falseNegative() + falsePositive());
    }
	
	double F1() const
	{
        auto pr = precision();
        auto re = recall();
		if (re + pr == 0)
		{
            return 0;
		}
        return (2 * pr * re) / (pr + re);
	}

	double precision() const
	{
        if (truePositive() + falsePositive() == 0)
        {
            return 0;
        }
        return static_cast<double>(truePositive()) / static_cast<double>(truePositive() + falsePositive());
	}

	double recall() const
	{
        if (truePositive() + falseNegative() == 0)
        {
            return 0;
        }
        return static_cast<double>(truePositive()) / static_cast<double>(truePositive() + falseNegative());
	}

	double MCC() const
    {
        auto tp = static_cast<double>(truePositive());
        auto tn = static_cast<double>(trueNegative());
        auto fp = static_cast<double>(falsePositive());
        auto fn = static_cast<double>(falseNegative());

        if (std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) == 0)
            return 0;
        return ((tp * tn) - (fp * fn)) / std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    }

    friend std::ostream& operator<<(std::ostream& out, const ConfusionMatrix& matrix);

	std::string to_string() const {
		std::ostringstream ss;
		ss << *this;
		return ss.str();
	}

    uint32_t truePositive() const;
    uint32_t trueNegative() const;
    uint32_t falsePositive() const;
    uint32_t falseNegative() const;

private:
    static const unsigned int m_sizeOfBinaryConfusionMatrix = 2u;
    /*               True condtion 
     *               0   1
     *  predicted  0 TN  FN
     *             1 FP  TP
     */
    std::array<std::array<uint32_t, m_sizeOfBinaryConfusionMatrix>, m_sizeOfBinaryConfusionMatrix> m_matrix;
};

inline uint32_t ConfusionMatrix::truePositive() const
{
    return m_matrix[1][1];
}

inline uint32_t ConfusionMatrix::trueNegative() const
{
    return m_matrix[0][0];
}

inline uint32_t ConfusionMatrix::falsePositive() const
{
    return m_matrix[1][0];
}

inline uint32_t ConfusionMatrix::falseNegative() const
{
    return m_matrix[0][1];
}


class ConfusionMatrixMulticlass
{
public:
    ConfusionMatrixMulticlass(const BaseSvmChromosome& individual,
        const dataset::Dataset<std::vector<float>, float, void>& testSamples);

    ConfusionMatrixMulticlass(phd::svm::ISvm& svm,
        const dataset::Dataset<std::vector<float>, float, void>& testSamples);

    explicit ConfusionMatrixMulticlass(std::vector<std::vector<uint32_t>> matrix);

    friend std::ostream& operator<<(std::ostream& out, const ConfusionMatrixMulticlass& matrix);

    std::string to_string() const {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    const std::vector<std::vector<uint32_t>>& getMatrix() const
    {
        return m_matrix;
    }

   /* inline uint32_t truePositive() const
    {
        return m_matrix[1][1];
    }

    inline uint32_t trueNegative() const
    {
        return m_matrix[0][0];
    }

    inline uint32_t falsePositive() const
    {
        return m_matrix[1][0];
    }

    inline uint32_t falseNegative() const
    {
        return m_matrix[0][1];
    }*/
	
private:
    unsigned int m_numberOfClasses;
    std::vector<std::vector<uint32_t>> m_matrix;
};
} // namespace svmComponents
