#include <iostream>
#include <memory>

#include "libPlatform/Subtree.h"
#include "libRandom/MersenneTwister64Rng.h"
#include "libDataset/CsvReader.h"

int main(int, char**){

    platform::Subtree s;
    auto rngEngine = std::make_unique<random::MersenneTwister64Rng>(0);

    phd::data::readCsv("file.csv");

    std::cout << "Hello, from SmartSVM!\n";
}
