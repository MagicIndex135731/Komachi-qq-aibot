"""Pure image-request recognition shared by group and private chat."""

import re

GROUP_IMAGE_REQUEST_FLAGS = re.IGNORECASE | re.DOTALL


GROUP_IMAGE_REQUEST_PATTERNS = (
    re.compile(
        r"^(?:请|麻烦|拜托)?(?:帮我)?画(?:一张|张)?(?:图片|图像|图)[\s,，。.!?？；;:：]*(?P<prompt>.+)$",
        GROUP_IMAGE_REQUEST_FLAGS,
    ),
    re.compile(
        r"^(?:请|麻烦|拜托)?(?:帮我)?画(?:个|一张|张)?(?P<prompt>.+)$", GROUP_IMAGE_REQUEST_FLAGS
    ),
    re.compile(r"^(?:请|麻烦|拜托)?(?:帮我)?来张(?P<prompt>.+)$", GROUP_IMAGE_REQUEST_FLAGS),
    re.compile(r"^(?:请|麻烦|拜托)?(?:帮我)?出图(?P<prompt>.+)$", GROUP_IMAGE_REQUEST_FLAGS),
    re.compile(
        r"^(?:请|麻烦|拜托)?(?:帮我)?生成(?:一张)?(?:图片|图像|图)?(?P<prompt>.+)$",
        GROUP_IMAGE_REQUEST_FLAGS,
    ),
)


GROUP_IMAGE_NEGATIVE_PATTERNS = (
    re.compile(r"会画图吗|能画图吗|会不会画图", re.IGNORECASE),
    re.compile(r"为什么.*出图", re.IGNORECASE),
    re.compile(r"谁画的", re.IGNORECASE),
    re.compile(r"识图", re.IGNORECASE),
)


GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS = (
    "改成",
    "换成",
    "变成",
    "做成",
    "转成",
    "替换成",
    "替换为",
)


GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS = (
    "模仿",
    "参考",
    "参照",
    "仿照",
    "照着",
    "按照",
    "按这张",
    "按这个",
    "按这两张",
    "根据这张",
    "根据这个",
    "根据这两张",
    "根据之前生成的图",
    "根据前面生成的图",
    "基于这张",
    "基于这个",
    "基于这两张",
    "基于前面生成的图",
    "同样动作",
    "同款动作",
    "同样画风",
    "同款画风",
    "同样构图",
    "同款构图",
    "在这张图基础上",
    "在这个图基础上",
    "在这两张图基础上",
    "在之前生成的图基础上",
    "在前面生成的图基础上",
    "这张图基础上",
    "这个图基础上",
    "这两张图基础上",
    "前图基础上",
    "前面那张图基础上",
)


GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS = (
    "图片",
    "图",
    "画",
    "画一张",
    "来一张",
    "来张",
    "出图",
    "生成",
    "做一张",
    "整一张",
    "搞一张",
    "弄一张",
)


LOOKUP_NORMALIZER = re.compile(
    r"[\s\u3000`~!@#$%^&*()_+\-=\[\]{}\\|;:'\",<.>/?，。！？：；、“”‘’（）《》【】]"
)


AUTO_WEB_REFERENCE_QUERY_PATTERN = re.compile(
    r"(?:先)?(?:去)?(?:网上|上网|联网)?(?:找|搜一下|搜索一下|搜索|搜)(?P<query>.+?)(?:的人设图|人设图|设定图|参考图|的图片|图片|的照片|照片|的图像|图像)",
    re.IGNORECASE,
)


AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN = re.compile(r"^(?:然后|再|并且|并|再去|接着|随后)+")
