load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "fl_agent",
    hdrs = ["fl/fl_agent.h"],
    srcs = ["fl/fl_agent.cpp"],
    deps = [
        "@arrayfire//:arrayfire",
        "@oneDNN//:oneDNN",
        "@flashlight//:flashlight",
    ],
    linkstatic=True,
    visibility = ["//visibility:public"],
)

exports_files(
    [
        "bazel-out/k8-fastbuild/bin/external/oneDNN/oneDNN/lib/libdnnl.so",
        "bazel-out/k8-fastbuild/bin/external/oneDNN/oneDNN/lib/libdnnl.so.2",
    ]
)