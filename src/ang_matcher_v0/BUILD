load("@rules_cc//cc:defs.bzl", "cc_library")


cc_library(
    name = "aux",
    hdrs = ["aux.h"],
    srcs = ["aux.cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "camera",
    hdrs = ["camera.h"],
    srcs = ["camera.cpp"],
    deps = [
        "@opencv//:opencv",
        ":aux",
        "//third_party/nlohmann:json",
    ],
    linkstatic=True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ang_matcher",
    hdrs = ["ang_matcher.h"],
    srcs = ["ang_matcher.cpp"],
    deps = [
        "@opencv//:opencv",
        ":aux",
    ],
    linkstatic=True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        ":aux",
        "//third_party/nlohmann:json",
        ":camera",
        ":ang_matcher",
        "@opencv//:opencv",
    ],
    includes = ["src"],
    data = glob(["images/**"]),
    #linkstatic=True,
    visibility = ["//visibility:public"],
)

