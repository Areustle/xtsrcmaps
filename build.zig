const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const top_level_path = b.path("");

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

    const xtsrcmaps = b.addSharedLibrary(.{
        .name = "xtsrcmaps",
        .target = target,
        .optimize = optimize,
        .pic = true,
    });
    xtsrcmaps.linkSystemLibrary("cxxopts");
    xtsrcmaps.linkSystemLibrary("fmt");
    xtsrcmaps.linkLibC();
    xtsrcmaps.linkLibCpp();
    xtsrcmaps.addCSourceFiles(.{ .files = &.{"xtsrcmaps/cli/cli.cxx"}, .flags = &cxxflags });
    xtsrcmaps.addIncludePath(top_level_path);

    b.installArtifact(xtsrcmaps);
}
