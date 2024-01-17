
#pragma once
#include "libPlatform/Subtree.h"

/* @wdudzik 
 *
 * IMPORTANT: All random number genarator in default are seeded by value of 0 (not random) 
 *            so to run any experiment it should be changed in all algorithms. Setting name is:
 *            --RandomNumberGenerator.isSeedRandom = true;
 *
 */
namespace genetic
{
/* @wdudzik This default config assume that input file are in the same folder as exe file (application)
            named as "test.csv", "trening.csv", "validation.csv"    
*/
class DefaultSvmConfig
{
public:
    static platform::Subtree getDefault();
    static platform::Subtree getRegressionSvr();
};

class DefaultKernelEvolutionConfig
{
public:
    static platform::Subtree getDefault();
    static platform::Subtree getRegressionSvr();
};

class DefaultGaSvmConfig
{
public:
    static platform::Subtree getDefault();
    static platform::Subtree getRegressionSvr();
};

class DefaultMemeticConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultFeatureSelectionConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultFeaturesMemeticConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultAlgaConfig
{
public:
    static platform::Subtree getDefault();
    static platform::Subtree getALMA();
    static platform::Subtree getALGA_regression();
};

class DefaultKTFConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultTFConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultFTConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultGridSearchConfig
{
public:
    static platform::Subtree getDefault();
    static const std::vector<std::string>& getAllGridsNames();
};

class DefaultSSVMConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultRandomSearchConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultRandomSearchInitPopConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultRandomSearchEvoHelpConfig
{
public:
    static platform::Subtree getDefault();
};

class CustomKernelConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultSequentialGammaConfig
{
public:
	static platform::Subtree getDefault();
};

class DefaultMultipleGammaMASVMConfig
{
public:
	static platform::Subtree getDefault();
};

class DefaultRbfLinearConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultSequentialGammaWithFeatureSelectionConfig
{
public:
    static platform::Subtree getDefault();
};


class DefaultEnsembleConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultEnsembleTreeConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultSESVMCorrectedConfig
{
public:
    static platform::Subtree getDefault();
};

class DefaultBigSetsEnsembleConfig
{
public:
    static platform::Subtree getDefault();
};
} // namespace genetic
