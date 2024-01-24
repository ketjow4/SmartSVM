from setuptools import setup, Extension, find_packages, Distribution
from setuptools.command.build_ext import build_ext
import pybind11


import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DCMAKE_CXX_STANDARD=17",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


cpp_args = ['/std:c++latest', '/WX-', '/permissive-',  '/wd4251', '/wd4275', '/wd4503', '/wd4840', '/FC', '/errorReport:prompt', '/GR',
            '/Zi', '/Gm-', '/O2', '/sdl', #remember change /Od to O2 for optimization, now only for debug purposes
            '/Zc:inline', '/fp:precise', '/openmp', '/GS', '/Zc:twoPhase-', '/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS'
            ]

link_args = [
    #'/LTCG:incremental',
    '/NXCOMPAT', '/DEBUG', '/MACHINE:X64', '/OPT:REF', '/OPT:ICF', '/WX:NO'
    ]

   

root = f'D:\\Deeva_PHD\\Deeva\\trunk\\src\\DeevaSvm'

libs = f'D:\\Deeva_PHD\\Deeva\\trunk\\src\\DeevaSvm\\packages'

boost = f'{libs}\\boost.1.66.0.0\\lib\\native\\include'
opencv = f'{libs}\\OpenCV.3.3.1\\include'
eigen = f'{libs}\\Eigen.3.3.4\\include'
gsl = f'{libs}\\Microsoft.Gsl.0.1.2.2\\build\\native\\include'

boost_po_bin = f'{libs}\\boost_program_options-vc141.1.66.0.0\\lib\\native'
boost_fs_bin = f'{libs}\\boost_filesystem-vc141.1.66.0.0\\lib\\native'
boost_system_bin = f'{libs}\\boost_system-vc141.1.66.0.0\\lib\\native'

openc_core_bin = f'{libs}\\OpenCV-Core.3.3.1\\lib\\native\\lib\\x64'
openc_hg_bin = f'{libs}\\OpenCV-Highgui.3.3.1\\lib\\native\\lib\\x64'
openc_imgc_bin = f'{libs}\\OpenCV-Imgcodecs.3.3.1\\lib\\native\\lib\\x64'
openc_imgp_bin = f'{libs}\\OpenCV-Imgproc.3.3.1\\lib\\native\\lib\\x64'
openc_ml_bin = f'{libs}\\OpenCV-Ml.3.3.1\\lib\\native\\lib\\x64'
openc_v_bin = f'{libs}\\OpenCV-Video.3.3.1\\lib\\native\\lib\\x64'
openc_vio_bin = f'{libs}\\OpenCV-Videoio.3.3.1\\lib\\native\\lib\\x64'

#RELEASE
other_libs = f'{root}\\bin\\x64\\Release'

#DEBUG
#other_libs = f'{root}\\bin\\x64\\Debug'

# python_folders = f'{root}\\..\\..\\extern\\python37'
# python_libs_folder = f'{root}\\..\\..\\extern\\python37\\libs'

python_folders = f'C:/anaconda3/'
python_libs_folder = f'C:/anaconda3/libs'

sfc_module = CMakeExtension(
    'DeevaPythonPackage',
    #sources=['module.cpp', 'DatasetLoader.cpp'],
    sourcedir=".",
#     include_dirs=[pybind11.get_include(),
#                  f'{root}\\platform',
#                  f'{root}',
#                  f'{root}\\app',
#                  f'{root}\\data',
#                  f'{root}\\genetic',
#                  f'{root}\\svm',
#                  boost,
#                  opencv,
#                  eigen,
#                  gsl
#                  ],
#     library_dirs=[boost_po_bin,
#                   boost_fs_bin,
#                   boost_system_bin,
#                   openc_core_bin,              
#                   openc_hg_bin,
#                   openc_imgc_bin,
#                   openc_imgp_bin,
#                   openc_ml_bin,
#                   openc_v_bin,
#                   openc_vio_bin,
#                   other_libs,
#                   python_libs_folder
#                   ],
#    libraries = [ 
#                   f'libGeneticComponents',
#                   f'libGeneticStrategies',
#                   f'libGeneticSvm',
#                   #f'libDataProvider',
#                   f'libDataset',
#                   #f'libException',
#                   #f'libFileSystem',
#                   f'libPlatform',
#                   f'libRandom',
#                   f'libStrategies',
#                   #f'libTime',
#                   f'libSvmComponents',
#                   #r'libSvmSigmoidTrain',
#                   f'libSvmStrategies',
#                   f'libSvm',
#                   #f'libLogger',
#                   f'opencv_core331',
#                   f'opencv_ml331',
#                   f'opencv_highgui331',
#                   f'opencv_imgcodecs331',
#                   f'opencv_imgproc331',
#                   ],

#     language='c++',
#    extra_compile_args=cpp_args,
#    extra_link_args =link_args,
    )

dll_path = 'dlls\\'

data_files = [
    #f'{dll_path}libLogger.dll',
    f'{dll_path}opencv_core331.dll',
    f'{dll_path}opencv_ml331.dll',
    f'{dll_path}opencv_highgui331.dll',
    f'{dll_path}opencv_imgcodecs331.dll',
    f'{dll_path}opencv_imgproc331.dll',
    f'{dll_path}opencv_video331.dll',
    f'{dll_path}opencv_videoio331.dll',
              ]

setup(
    author = 'Wojciech Dudzik',
    name='DeevaPythonPackage',
    version='0.3.7',
    description='Python package with Evolutionary SVM C++ code (PyBind11)',
    ext_modules=[sfc_module],

    cmdclass={"build_ext": CMakeBuild},
    #srcipts=['mutualInfo.py'],
    packages=find_packages(),
    #include_package_data=True,
    #package_data={
    #    '': ['*.dll'],
    #},
    #package_dir={"DeevaPythonPackage": "DeevaPythonPackage"},
    data_files=data_files,
    python_requires='>3.7.0',
    #distclass=BinaryDistribution
)

#D:\\Deeva_PHD\\Deeva\\trunk\\src\\DeevaSvm\\app\\DeevaPythonModule\\dlls\\*.dll'