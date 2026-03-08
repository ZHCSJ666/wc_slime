# wc_slime

本仓库用于使用 **slime 框架**训练 **Qwen3-VL-8B-Instruct** 模型，任务为 **WC VQA RL 训练**。

训练所使用的组件：

* 模型：Qwen3-VL-8B-Instruct
* 数据集：自动从 HuggingFace 下载
  ZHCSJ/wc-en-open-slime-4k
* Reward 函数：reward/wc_reward.py

运行训练时，只需要在仓库目录下执行：

```bash
bash run_wc_vlm.sh
```

脚本会自动完成以下步骤：

* 从 HuggingFace 下载数据集
* 将数据集转换为 train.parquet
* 加载模型 Qwen3-VL-8B-Instruct
* 启动 slime 训练

无需手动准备数据集或数据格式。
模型也会自动下载
训练结果会保存到以下路径：

```
/data/oss_bucket_0/users/xintong/team/longqin/outputs/wc_slime/wc_vlm_Qwen3-VL-8B-Instruct
```

训练日志在：

```
/data/oss_bucket_0/users/xintong/team/longqin/outputs/wc_slime/wc_vlm_Qwen3-VL-8B-Instruct/logs
```

如果脚本运行失败，请把 **log 文件发给我**，我会根据 log 进行排查。


