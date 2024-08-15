# llvm-toolchain.cmake

# Specify the Homebrew LLVM installation path
set(HOMEBREW_LLVM_PATH /opt/homebrew/opt/llvm)

# Specify the C and C++ compilers
set(CMAKE_C_COMPILER ${HOMEBREW_LLVM_PATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${HOMEBREW_LLVM_PATH}/bin/clang++)

# Specify the path to LLVM's include and library directories
set(CMAKE_INCLUDE_PATH ${HOMEBREW_LLVM_PATH}/include)
set(CMAKE_LIBRARY_PATH ${HOMEBREW_LLVM_PATH}/lib)

# Set the linker flags to use LLVM's library path
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${HOMEBREW_LLVM_PATH}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${HOMEBREW_LLVM_PATH}/lib")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -L${HOMEBREW_LLVM_PATH}/lib")

# Add the LLVM bin directory to the PATH
set(CMAKE_PREFIX_PATH ${HOMEBREW_LLVM_PATH})
set(CMAKE_FIND_ROOT_PATH ${HOMEBREW_LLVM_PATH})
set(CMAKE_SYSTEM_PREFIX_PATH ${HOMEBREW_LLVM_PATH})

# If needed, specify the compiler and linker for the CMake try_compile checks
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
