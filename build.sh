#!/bin/zsh
# Builds libfencepost.dylib and the test program, both universal (arm64 + x86_64: the
# x86_64 slice is the one Wine under Rosetta loads), signed ad hoc.
#   ./build.sh                 build into out/
#   ./build.sh release 0.1.0   also out/fencepost-0.1.0-macos.tar.gz: the dylib without debug
#                              information (it holds local paths), README.md, LICENSE
set -e
cd "${0:A:h}"
mkdir -p out
FLAGS=(-arch arm64 -arch x86_64 -O2 -g -Wall -Wextra -Wno-unused-parameter -fno-objc-arc -mmacosx-version-min=13.0)
# Build to a temporary name and rename it over: a running program that has the old dylib
# mapped (a Steam session) keeps its copy intact, instead of hanging on a rewritten file.
clang $FLAGS -dynamiclib -install_name @rpath/libfencepost.dylib -o out/.libfencepost.dylib.new src/fencepost.m \
    -framework Metal -framework Foundation
clang $FLAGS -o out/fptest tests/fptest.m -framework Metal -framework Foundation
codesign -f -s - out/.libfencepost.dylib.new out/fptest 2>/dev/null
mv -f out/.libfencepost.dylib.new out/libfencepost.dylib
echo "built out/libfencepost.dylib and out/fptest"

if [[ $1 == release ]]; then
    v=${2:?usage: ./build.sh release VERSION}
    d=out/fencepost-$v-macos
    rm -rf $d && mkdir -p $d
    cp out/libfencepost.dylib README.md LICENSE $d/
    strip -S -x $d/libfencepost.dylib 2>/dev/null   # re-signed below
    codesign -f -s - $d/libfencepost.dylib 2>/dev/null
    tar -czf $d.tar.gz -C out fencepost-$v-macos
    shasum -a 256 $d.tar.gz | sed 's|out/||' > $d.tar.gz.sha256
    echo "built $d.tar.gz"
fi
