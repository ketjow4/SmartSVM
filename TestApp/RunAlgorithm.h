#pragma once

#include <string>
//#include "AppUtils/AppUtils.h"
#include "AppUtils/PythonFeatureSelection.h"

#include "Commons.h"



void RunAlgorithm(int argc, char* argv[], AlgorithmName algorithmToRun);

std::map<std::string, testApp::DatasetInfo> getDatasetInformations(const testApp::configTestApp& config);

void RunRandomSearchExperiments(std::vector<std::string> dataFolders);

void RunAlgaExperiments(std::vector<std::string> dataFolders);

void RunMasvmExperiments(std::vector<std::string> dataFolders);

void RunGasvmExperiments(std::vector<std::string> dataFolders);

void RunFSALMAExperiments(std::vector<std::string> dataFolders);

void RunFSALMA(int argc, char* argv[]);

void RunGridSearch(std::map<uint32_t, KernelParams>& gridSearchResults, std::vector<std::string> dataFolders);

void RunGsNoFeatureSelection(int argc, char* argv[]);

void RunGridSearchWithFeatureSelection(std::map<uint32_t, KernelParams>& gridSearchResults, std::vector<std::string> dataFolders);

void RunSsvmGecco2019(std::vector<std::string> dataFolders);

void RunEnsembleExperiments(std::vector<std::string> dataFolders);

void RunSingleAlgorithmFS(std::vector<std::string> dataFolders, std::vector<std::string> filters);