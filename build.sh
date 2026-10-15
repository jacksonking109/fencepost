#!/bin/zsh
# Builds libfencepost.dylib and the test program, both universal (arm64 + x86_64: the
# x86_64 slice is the one Wine under Rosetta loads), signed ad hoc.
set -e
cd "${0:A:h}"
mkdir -p out
FLAGS=(-arch arm64 -arch x86_64 -O2 -g -Wall -Wextra -Wno-unused-parameter -fno-objc-arc -mmacosx-version-min=13.0)
# Build to a temporary name and rename it over: a running program that has the old dylib
# mapped (a Steam session) keeps its copy intact, instead of hanging on a rewritten file.
clang $FLAGS -dynamiclib -o out/.libfencepost.dylib.new src/fencepost.m -framework Metal -framework Foundation
clang $FLAGS -o out/fptest tests/fptest.m -framework Metal -framework Foundation
codesign -f -s - out/.libfencepost.dylib.new out/fptest 2>/dev/null
mv -f out/.libfencepost.dylib.new out/libfencepost.dylib
echo "built out/libfencepost.dylib and out/fptest"
