# Xtsrcmaps: An Experimental Fermi Gtsrcmaps replacement

**Beta Software**

Currently implements support for point source models on the WCS projection. 

Planned Future features:
 - Energy Dispersion (EDISP)
 - Healpix Projection
 - Composite Sources
 - Extended Sources
 - Diffuse Sources
 - CALDB IRFs

## Compiling from Source

### Requirements
- A C++ compiler supporting the C++/20 standard or later
- CMake Version 3.21 or later
- Installed library: cfitsio
- Installed library: wcslib

### Compilation steps

1. Acquire the code:
``` bash
https://github.com/Areustle/xtsrcmaps.git
cd xtsrcmaps
```

2. Configure with CMake. Optionally use CMake specific features to target
compilers and build systems.
``` bash
cmake   -S . \\
        -B Release \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \\
        -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \\
        -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm \\
        -DCMAKE_INSTALL_PREFIX=~/mycode/
        -DCMAKE_EXPORT_COMPILE_COMMANDS=On \\
        -G Ninja
```

3. Build the code.
```bash
cmake --build Release
```

