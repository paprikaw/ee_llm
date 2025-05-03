#!/usr/bin/env python3
"""
run_server.py

启动微服务：
  STAGE=0 时做 Stage0；STAGE=1 时做 Stage1
STAGE1_REPLICAS 环境变量控制 Stage0 转发到哪些 Stage1 实例。
"""
import os
import sys
import torch
from flask import Flask, request, jsonify
import requests

# Appendix: 将 Megatron 根目录加入 PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from megatron import get_args, print_rank_0
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.arguments import core_transformer_config_from_args
from megatron.model import EarlyExitGPTModel

import api  # 下面我们定义的 api.py

def model_provider(pre_process=True, post_process=True):
    config = core_transformer_config_from_args(get_args())
    print_rank_0("Building EarlyExitGPT model block...")
    return EarlyExitGPTModel(config,
                             num_tokentypes=0,
                             parallel_output=False,
                             pre_process=pre_process,
                             post_process=post_process)

def add_text_generate_args(parser):
    group = parser.add_argument_group(title="text generation")
    group.add_argument("--port", type=int, default=5000,
                       help="HTTP port for this stage")
    return parser

def main():
    # 1) 初始化 Megatron（内部可能做 NCCL 初始化，但我们不用于跨阶段）
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={
                            "tokenizer_type": "GPT2BPETokenizer",
                            "no_load_rng": True,
                            "no_load_optim": True,
                        })
    args = get_args()
    print_rank_0(f"[ENV] STAGE={os.environ.get('STAGE','0')}  PORT={args.port}")

    # 2) 构建模型并加载 checkpoint
    model_list = get_model(model_provider, wrap_with_ddp=False)
    model = model_list[0]
    if args.load:
        _ = load_checkpoint(model, None, None)
    model.eval()

    # 3) 根据 STAGE 启动不同的 Flask app
    stage = os.environ.get("STAGE", "0")
    port  = args.port

    if stage == "0":
        # ——————— Stage 0 ———————
        app = Flask(__name__)
        replicas = os.environ.get("STAGE1_REPLICAS", "")
        # 格式: http://host1:port,http://host2:port
        replicas = [r.rstrip("/") for r in replicas.split(",") if r]
        from itertools import cycle
        rr = cycle(replicas)

        @app.route("/generate", methods=["POST"])
        def generate_stage0():
            payload = request.get_json()
            prompt  = payload.get("prompt", "")
            gen_args= payload.get("gen_args", {})
            # ① 本地执行第一段推理，返回一个 activation 包
            act_payload = api.stage0_infer(model, prompt, **gen_args)
            # ② 轮询转发给一个 Stage1 副本
            target = next(rr)
            resp   = requests.post(f"{target}/generate", json=act_payload, timeout=30)
            return jsonify(resp.json())

        app.run(host="0.0.0.0", port=port, threaded=True)

    else:
        # ——————— Stage 1 ———————
        app = Flask(__name__)

        @app.route("/generate", methods=["POST"])
        def generate_stage1():
            payload = request.get_json()
            # ① 由 api.py 处理 activation 包，执行第二段推理
            result = api.stage1_infer(model, payload)
            return jsonify(result)

        app.run(host="0.0.0.0", port=port, threaded=True)

if __name__ == "__main__":
    main()
