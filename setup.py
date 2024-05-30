from setuptools import setup, Extension, find_packages, Distribution
from setuptools.command.build_ext import build_ext
import pybind11
import pybind11_stubgen


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

        #debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        #cfg = "Debug" if debug else "Release"
        cfg = "RelWithDebInfo"
        print(f'Configuration: {cfg}')

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

        cmake_args += [f"-DVERSION_INFO={self.distribution.get_version()}"]

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
            else:
                build_args += [f"-j 8"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        print(f'Running cmake build:  cmake --build . {build_args}')
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )
        
        # Generate stubs after building the extension
        self.generate_stubs(ext.name, extdir)

    def generate_stubs(self, module_name, output_dir):
        # Ensure the module can be imported
        try:
            import importlib
            import sys

            # Adding the directory containing the built module to sys.path
            sys.path.insert(0, str(output_dir))
            importlib.import_module(module_name)
            sys.path.pop(0)
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
            raise


        # import importlib.util
        # import sys
        # from pathlib import Path

        # def import_so_module(module_name, file_path):
        #     # Convert file path to absolute path
        #     file_path = Path(file_path).resolve()
            
        #     # Create a module spec from the file location
        #     spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            
        #     # Create a module from the spec
        #     module = importlib.util.module_from_spec(spec)
            
        #     # Add the module to sys.modules
        #     sys.modules[module_name] = module
            
        #     # Execute the module in its own namespace
        #     spec.loader.exec_module(module)
            
        #     return module
        
        # module = import_so_module("DeevaPythonPackage", str(output_dir / "DeevaPythonPackage.cpython-39-darwin.so"))


        # Run pybind11-stubgen programmatically
        print(f"Generating stubs for module: {module_name}")

        ### Seems like ugly hack to make it work
        from pybind11_stubgen import stub_parser_from_args, Printer, Writer, to_output_and_subdir, run, arg_parser

        args = arg_parser().parse_args(["DeevaPythonPackage"])
        
        parser = stub_parser_from_args(args)
        printer = Printer(invalid_expr_as_ellipses=True)

        out_dir, sub_dir = to_output_and_subdir(
            output_dir="DeevaPythonPackage-stubs",
            module_name=module_name,
            root_suffix=None,
        )

        run(parser,
            printer,
            module_name,
            out_dir,
            sub_dir=sub_dir,
            dry_run=False,
            writer=Writer(stub_ext="pyi"))

        

    

# cpp_args = ['/std:c++latest', '/WX-', '/permissive-',  '/wd4251', '/wd4275', '/wd4503', '/wd4840', '/FC', '/errorReport:prompt', '/GR',
#             '/Zi', '/Gm-', '/O2', '/sdl', #remember change /Od to O2 for optimization, now only for debug purposes
#             '/Zc:inline', '/fp:precise', '/openmp', '/GS', '/Zc:twoPhase-', '/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS'
#             ]

# link_args = [
#     #'/LTCG:incremental',
#     '/NXCOMPAT', '/DEBUG', '/MACHINE:X64', '/OPT:REF', '/OPT:ICF', '/WX:NO'
#     ]

#pybind11.get_include()

cpp_svm_module = CMakeExtension(
    'DeevaPythonPackage',
    sourcedir=".",
    )


setup(
    author = 'Wojciech Dudzik',
    name='DeevaPythonPackage',
    version='0.3.11',
    description='Python package with Evolutionary SVM C++ code (PyBind11)',

    setup_requires=['pybind11-stubgen'],

    ext_modules=[cpp_svm_module],
    cmdclass={"build_ext": CMakeBuild},
    #srcipts=['mutualInfo.py'],
    # packages=find_packages(),
    # package_dir={'': '.'},
    # packages=["DeevaPythonPackage-stubs", "DeevaPackage"],
    # include_package_data=True,
    # # DeevaPythonPackage-stubs contains type hint data in a .pyi file, per PEP 561
    # package_data={
    #     "DeevaPythonPackage-stubs": ["*.pyi"],
    # },

    package_dir={'': '.'},
    packages=["DeevaPythonPackage-stubs", "DeevaPackage"],
    include_package_data=True,
    # DeevaPythonPackage-stubs contains type hint data in a .pyi file, per PEP 561
    package_data={
        "DeevaPythonPackage-stubs": ["*.pyi"],
    },

    
    # data_files=data_files,
    python_requires='>3.7.0',
    
    #distclass=BinaryDistribution
     # Add this to enable in-place build for development
    zip_safe=False,
)