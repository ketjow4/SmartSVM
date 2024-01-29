#pragma once

#include <filesystem>
#include <vector>


std::vector<bool> runFeatureSelection(std::filesystem::path treningSetPath);

std::vector<double> runMutualInfo(std::filesystem::path treningSetPath);

