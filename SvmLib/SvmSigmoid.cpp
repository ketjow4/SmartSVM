//Copyright(c) 2000 - 2014 Chih - Chung Chang and Chih - Jen Lin
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions
//are met :
//
//1. Redistributions of source code must retain the above copyright
//notice, this list of conditions and the following disclaimer.
//
//2. Redistributions in binary form must reproduce the above copyright
//notice, this list of conditions and the following disclaimer in the
//documentation and / or other materials provided with the distribution.
//
//3. Neither name of copyright holders nor the names of its contributors
//may be used to endorse or promote products derived from this software
//without specific prior written permission.
//
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE REGENTS OR
//CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
//        NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <vector>
#include "SvmSigmoid.h"

namespace libSvm {namespace utils
{
SigmoidParameters::SigmoidParameters(double a, double b)
    : m_A(a)
    , m_B(b)
{
}

// @wdudzik This code is taken from libSVM. Changed style to be more consistent with phd
// Changed raw pointers to std::vector
SigmoidParameters sigmoidTrain(unsigned int datasetSize,
                               gsl::span<const float> decisionValues,
                               gsl::span<const int> labels)
{
    double prior1 = 0, prior0 = 0;
    unsigned int i;

    for (i = 0; i < datasetSize; i++)
    {
        if (labels[i] > 0)
        {
            prior1 += 1;
        }
        else
        {
            prior0 += 1;
        }
    }

    constexpr int maxIter = 100; // Maximal number of iterations
    constexpr double minStep = 1e-10; // Minimal step taken in line search
    constexpr double sigma = 1e-12; // For numerically strict PD of Hessian
    constexpr double epsilon = 1e-5;
    double highTarget = (prior1 + 1.0) / (prior1 + 2.0);
    double lowTarget = 1 / (prior0 + 2.0);
    std::vector<double> t(datasetSize);
    double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
    double newA, newB, newf, d1, d2;
    int iter;

    // Initial Point and Initial Fun Value
    double sigmoidA = 0.0;
    double sigmoidB = log((prior0 + 1.0) / (prior1 + 1.0));
    double fval = 0.0;

    for (i = 0; i < datasetSize; i++)
    {
        if (labels[i] > 0)
        {
            t[i] = highTarget;
        }
        else
        {
            t[i] = lowTarget;
        }
        fApB = decisionValues[i] * sigmoidA + sigmoidB;
        if (fApB >= 0)
        {
            fval += t[i] * fApB + log(1 + exp(-fApB));
        }
        else
        {
            fval += (t[i] - 1) * fApB + log(1 + exp(fApB));
        }
    }
    for (iter = 0; iter < maxIter; iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11 = sigma; // numerically ensures strict PD
        h22 = sigma;
        h21 = 0.0;
        g1 = 0.0;
        g2 = 0.0;
        for (i = 0; i < datasetSize; i++)
        {
            fApB = decisionValues[i] * sigmoidA + sigmoidB;
            if (fApB >= 0)
            {
                p = exp(-fApB) / (1.0 + exp(-fApB));
                q = 1.0 / (1.0 + exp(-fApB));
            }
            else
            {
                p = 1.0 / (1.0 + exp(fApB));
                q = exp(fApB) / (1.0 + exp(fApB));
            }
            d2 = p * q;
            h11 += decisionValues[i] * decisionValues[i] * d2;
            h22 += d2;
            h21 += decisionValues[i] * d2;
            d1 = t[i] - p;
            g1 += decisionValues[i] * d1;
            g2 += d1;
        }

        // Stopping Criteria
        if (fabs(g1) < epsilon && fabs(g2) < epsilon)
        {
            break;
        }

        // Finding Newton direction: -inv(H') * g
        det = h11 * h22 - h21 * h21;
        dA = -(h22 * g1 - h21 * g2) / det;
        dB = -(-h21 * g1 + h11 * g2) / det;
        gd = g1 * dA + g2 * dB;

        stepsize = 1; // Line Search
        while (stepsize >= minStep)
        {
            newA = sigmoidA + stepsize * dA;
            newB = sigmoidB + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i = 0; i < datasetSize; i++)
            {
                fApB = decisionValues[i] * newA + newB;
                if (fApB >= 0)
                {
                    newf += t[i] * fApB + log(1 + exp(-fApB));
                }
                else
                {
                    newf += (t[i] - 1) * fApB + log(1 + exp(fApB));
                }
            }
            // Check sufficient decrease
            if (newf < fval + 0.0001 * stepsize * gd)
            {
                sigmoidA = newA;
                sigmoidB = newB;
                fval = newf;
                break;
            }
            stepsize = stepsize / 2.0;
        }

        if (stepsize < minStep)
        {
            //info("Line search fails in two-class probability estimates\n");
            break;
        }
    }
    return SigmoidParameters(sigmoidA, sigmoidB);
}
}} // namespace libSvm::utils
