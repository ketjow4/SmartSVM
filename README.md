# SmartSVM


How to setup VCPKG manager --- Not needed anymore

https://thatonegamedev.com/cpp/how-to-manage-dependencies-with-cmake-and-vcpkg/

https://github.com/microsoft/vcpkg/tree/2023.12.12

OpenCV example
https://gist.github.com/UnaNancyOwen/5061d8c966178b753447e8a9f9ac8cf1

Usefull CMake resources
https://stackoverflow.com/questions/13703647/how-to-properly-add-include-directories-with-cmake


Fresh install script
git clone https://github.com/microsoft/vcpkg/tree/2023.12.12
./vcpkg/bootstrap-vcpkg.bat    or for linux/macos   ./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install opencv3 boost ms-gsl

//proper conda env to activate need only pybind11 to build
conda activate python310
python setup.py bdist_wheel

Do not run this
.\vcpkg install opencv[vtk]   --- warning this take long time (45-60 minutes to build)


## TODO list

- Add python test
- Fix visualizations
- Add wrapper for all relevant algorithms
    - SE-SVM fix feature selection at the start
    - CE-SVM not ready
    - ECE-SVM not ready
- CI build package on GHA
- Add our datasets --- Done (at least for 2D datasets), need to fix names
- Add examples
- Add python code for analysing the summaries (as used in Phd)
- Fix SE-SVM (feature selection init) and ECE-SVM (extra tree at the end)
- Publish to pip
- Create documentation
- Generate python stubs
- Add microbenchmarks 