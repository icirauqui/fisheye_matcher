load("@rules_cc//cc:defs.bzl", "cc_binary")


cc_binary(
    name = "main_ang_matcher_v0",
    srcs = ["src/main_ang_matcher_v0.cpp"],
    deps = [
        "//src/ang_matcher_v0:aux",
        "//third_party/nlohmann:json",
        "//src/ang_matcher_v0:camera",
        "//src/ang_matcher_v0:ang_matcher",
        "@opencv//:opencv",
    ],
    includes = [
        "src/ang_matcher_v0",
    ],
    data = glob(["images/**"]),
    #linkstatic=True,
    visibility = ["//visibility:public"],
)



cc_binary(
    name = "main_fe_lens_matcher",
    srcs = [
        "src/main_fe_lens_matcher.cpp",
        ],
    deps = [
        "//src/fe_lens:fe_lens",
        "//src/matcher:matcher",
        "//src/ang_matcher:ang_matcher",
        "@opencv//:opencv",
    ],
    includes = [
      "src",
    ],
    data = glob(["images/**"]),
    visibility = ["//visibility:public"],
)