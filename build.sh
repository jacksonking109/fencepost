#!/bin/zsh
# Builds libfencepost.dylib, the test programs and the trigger helpers, all universal (arm64 + x86_64: the
# x86_64 slice is the one Wine under Rosetta loads), signed ad hoc.
#   ./build.sh                 build into out/
#   ./build.sh release 0.3.0   also out/fencepost-0.3.0-macos.tar.gz: the dylib, fencepost-trigger and
#                              Fencepost Trigger.app without debug information (it holds local paths),
#                              README.md, LICENSE
set -e
cd "${0:A:h}"
mkdir -p out
FLAGS=(-arch arm64 -arch x86_64 -O2 -g -Wall -Wextra -Wno-unused-parameter -fno-objc-arc -mmacosx-version-min=13.0)
# Build to a temporary name and rename it over: a running program that has the old dylib
# mapped (a Steam session) keeps its copy intact, instead of hanging on a rewritten file.
clang $FLAGS -dynamiclib -install_name @rpath/libfencepost.dylib -o out/.libfencepost.dylib.new src/fencepost.m \
    -framework Metal -framework Foundation
if [[ $1 != release ]]; then
    clang $FLAGS -DFENCEPOST_TEST_HOOKS -dynamiclib -install_name @rpath/libfencepost-testhooks.dylib \
        -o out/.libfencepost-testhooks.dylib.new src/fencepost.m -framework Metal -framework Foundation
    codesign -f -s - out/.libfencepost-testhooks.dylib.new 2>/dev/null
    mv -f out/.libfencepost-testhooks.dylib.new out/libfencepost-testhooks.dylib
    echo "built out/libfencepost-testhooks.dylib"
fi
clang $FLAGS -o out/fptest tests/fptest.m -framework Metal -framework Foundation
clang $FLAGS -o out/tracetest tests/tracetest.m -framework Metal -framework Foundation
clang $FLAGS -o out/ringtest tests/ringtest.m -framework Metal -framework Foundation
clang $FLAGS -o out/capturetest tests/capturetest.m -framework Metal -framework Foundation
clang $FLAGS -fobjc-arc -o out/mtl4test tests/mtl4test.m -framework Metal -framework Foundation
clang $FLAGS -fobjc-arc -o out/mtl4bench tests/mtl4bench.m -framework Metal -framework Foundation
clang $FLAGS -fobjc-arc -o out/mtl4frames tests/mtl4frames.m -framework Metal -framework Foundation -framework QuartzCore
clang $FLAGS -fobjc-arc -o out/mtl4races tests/mtl4races.m -framework Metal -framework Foundation
clang $FLAGS -o out/fencepost-trigger tools/trigger/fencepost-trigger.c
app='out/Fencepost Trigger.app'
mkdir -p "$app/Contents/MacOS"
# Built outside the bundle so its debug symbols (out/fencepost-trigger-app.dSYM) stay out of the app.
clang $FLAGS -fobjc-arc -o out/fencepost-trigger-app tools/trigger/FencepostTrigger.m \
    -framework Cocoa -framework Carbon
rm -rf "$app/Contents/MacOS/Fencepost Trigger.dSYM"
mv -f out/fencepost-trigger-app "$app/Contents/MacOS/Fencepost Trigger"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>Fencepost Trigger</string>
    <key>CFBundleIdentifier</key><string>io.github.jacksonking109.fencepost-trigger</string>
    <key>CFBundleName</key><string>Fencepost Trigger</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST
codesign -f -s - out/.libfencepost.dylib.new out/fptest out/tracetest out/ringtest out/capturetest \
    out/mtl4test out/mtl4bench out/mtl4frames out/mtl4races 2>/dev/null
codesign -f -s - out/fencepost-trigger "$app" 2>/dev/null
mv -f out/.libfencepost.dylib.new out/libfencepost.dylib
echo "built out/libfencepost.dylib, out/fptest, out/tracetest, out/ringtest, out/capturetest and out/fencepost-trigger"
echo "built out/mtl4test, out/mtl4bench, out/mtl4frames and out/mtl4races"
echo "built $app"

if [[ $1 == release ]]; then
    v=${2:?usage: ./build.sh release VERSION}
    d=out/fencepost-$v-macos
    rm -rf $d && mkdir -p $d
    cp out/libfencepost.dylib out/fencepost-trigger README.md LICENSE $d/
    cp -R "$app" $d/
    rapp="$d/Fencepost Trigger.app"
    /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string $v" "$rapp/Contents/Info.plist"
    # Stripped binaries are re-signed below.
    strip -S -x $d/libfencepost.dylib $d/fencepost-trigger "$rapp/Contents/MacOS/Fencepost Trigger" 2>/dev/null
    codesign -f -s - $d/libfencepost.dylib $d/fencepost-trigger "$rapp" 2>/dev/null
    tar -czf $d.tar.gz -C out fencepost-$v-macos
    shasum -a 256 $d.tar.gz | sed 's|out/||' > $d.tar.gz.sha256
    echo "built $d.tar.gz"
fi
