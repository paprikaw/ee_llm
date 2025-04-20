# 1. 设置并行构建线程数
MAX_JOBS=24 python -m pip wheel flash-attn \
  --no-deps \
  --no-build-isolation \
  -w ./wheelhouse
