import os
import subprocess
from flask import Flask, request, jsonify, send_file
import requests


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 最大上传文件大小为100MB

# os.chdir("/home/chenzan/workSpace/yungeng/SaProt")


@app.route("/run_ray", methods=["POST"])
def run_model():
    data = request.json
    print(data)

    lr = data.get("lr")
    epoch = data.get("epoch")
    num_samples=data.get("num_samples")
    max_num_epochs=data.get("max_num_epochs")
    dropout = data.get("dropout")

    file_path = data.get("file_path")
    sequence = data.get("sequence")


    # 解析输入参数
    # seq = data.get("seq", "")
    # mut_info = data.get("mut_info", "")

    # 构建命令行参数
    cmd = [
        "python",
        "/home/chenzan/workSpace/yungeng/ray_esm/esm_ray.py",
        lr, epoch, num_samples, max_num_epochs, dropout, file_path, sequence, f" > ./{lr}_{epoch}_{dropout}_{sequence}.log "
    ]

    print(cmd)
    log_path = f'./{lr}_{epoch}_{dropout}_{sequence}.log'

    # 运行命令行程序
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print(stdout, process.returncode)
    stdout = str(stdout, encoding="utf-8")

    if process.returncode != 0:
        return jsonify({"error": stderr.decode()}), 500

    # 返回生成的文件
    # url = []
    # for root, dirs, files in os.walk(tmp_dir):
    #     for file in files:
    #         print(file)
    return jsonify({"log_path": log_path})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8007)

