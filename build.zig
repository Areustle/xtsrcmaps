const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions(.{});

    const flags = [_][]const u8{
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused-parameter",
        "-Wno-array-bounds",
        "-Wno-deprecated-enum-enum-conversion",
        "-Wno-unused-but-set-variable",
    };
    const cxxflags = flags ++ [_][]const u8{
        "-std=c++23",
    };

    const xtsrcmaps = b.addSharedLibrary("xtsrcmaps", null);
    xtsrcmaps.linkSystemLibrary("cxxopts");
    xtsrcmaps.linkSystemLibrary("fmt");
    xtsrcmaps.setTarget(target);
    xtsrcmaps.setBuildMode(mode);
    xtsrcmaps.linkLibC();
    xtsrcmaps.linkLibCpp();
    xtsrcmaps.force_pic = true;
    xtsrcmaps.addCSourceFiles(&.{
        "xtsrcmaps/cli/cli.cxx",
    }, &cxxflags);

    b.InstallArtifacts(xtsrcmaps);
}
