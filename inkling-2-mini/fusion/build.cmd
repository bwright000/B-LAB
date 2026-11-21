@echo off
rem Build llama.cpp (src\llama.cpp) for RTX 50-series only (sm_120), Release, CUDA 12.8 + MSVC 14.44 via Ninja.
rem Usage: build.cmd            (incremental rebuild after code edits is the same command)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cd /d "%~dp0llama.cpp" || exit /b 1
rem configure every time (cheap): CUDA sources are globbed, so new .cu files need a re-scan
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DGGML_NATIVE=ON ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=ON >nul || exit /b 1
cmake --build build --config Release -j 24 --target llama-server llama-bench llama-cli test-backend-ops llama-perplexity || exit /b 1
echo BUILD_OK
