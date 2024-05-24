
#include "MersenneTwister64Rng.h"

namespace my_random
{
MersenneTwister64Rng::MersenneTwister64Rng(uint64_t seed)
{
    m_engine = std::mt19937_64(seed);
}
} // namespace my_random

