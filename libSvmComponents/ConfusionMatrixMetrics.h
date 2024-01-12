#pragma once

#include "libSvmComponents/ConfusionMatrix.h"

namespace svmComponents
{
inline double Accuracy(const ConfusionMatrix& matrix)
{
    return static_cast<double>(matrix.trueNegative() + matrix.truePositive()) /
            static_cast<double>(matrix.falseNegative() + matrix.falsePositive() + matrix.truePositive() + matrix.trueNegative());
}


template <class T> struct innermost_type
{
    using type = T;
};

template <class T> struct innermost_type<std::vector<T>>
{
    using type = typename innermost_type<T>::type;
};

template <class T>
using innermost_type_t = typename innermost_type<T>::type;

template <class T>
auto sum_all(const std::vector<T>& nested_v)
{
    innermost_type_t<T> sum = 0;

    for (auto& e : nested_v)
    {
        if constexpr (std::is_arithmetic<T>::value)
            sum += e;
        else
            sum += sum_all(e);
    }

    return sum;
}



inline double Accuracy(const ConfusionMatrixMulticlass& matrix)
{
    auto values = matrix.getMatrix();
    auto all = sum_all(values);

    auto correct = 0;
	for(auto i = 0u; i < values.size(); ++i)
	{
        correct += values[i][i];
	}

    return static_cast<double>(correct) / static_cast<double>(all);
}

inline double MCC(const ConfusionMatrix& matrix)
{
    auto tp = static_cast<double>(matrix.truePositive());
    auto tn = static_cast<double>(matrix.trueNegative());
    auto fp = static_cast<double>(matrix.falsePositive());
    auto fn = static_cast<double>(matrix.falseNegative());

    if (std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) == 0)
        return 0;
    return ((tp * tn) - (fp * fn)) / std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
	
    /*if math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)) == 0:
    return 0
        return ((self.tp * self.tn) - (self.fp * self.fn)) / math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))*/
}

inline double BalancedAccuracy(const ConfusionMatrix& matrix)
{
    double tpr = static_cast<double>(matrix.truePositive()) / static_cast<double>(matrix.truePositive() + matrix.falseNegative());
    double tnr = static_cast<double>(matrix.trueNegative()) / static_cast<double>(matrix.trueNegative() + matrix.falsePositive());
	
    return (tpr + tnr) / 2;
}


inline double BalancedAccuracy(const ConfusionMatrixMulticlass& matrix)
{

    auto values = matrix.getMatrix();

    auto balancedAcc = 0.0;
	for(auto i = 0u; i < values.size(); ++i)
	{
        auto sum = static_cast<double>(sum_all(values[i]));
        if (sum != 0)
        {
            balancedAcc += static_cast<double>(values[i][i]) / sum;
        }
	}
    
    return balancedAcc / values.size();
}
} // namespace svmComponents
