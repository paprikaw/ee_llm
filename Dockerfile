FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG FLASH_ATTEN_WHEEL=flash_attn-2.7.0.post2-cp38-cp38-linux_x86_64.whl
WORKDIR /workspace/EE-LLM

# 拷贝项目源码
COPY . /workspace/EE-LLM

# 安装挂载进来的 wheel
RUN pip install "/workspace/wheelhouse/${FLASH_ATTEN_WHEEL}"
RUN pip install flask flask_restful sentencepiece

# 入口脚本等其它内容不变……
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


ENTRYPOINT ["/entrypoint.sh"]
