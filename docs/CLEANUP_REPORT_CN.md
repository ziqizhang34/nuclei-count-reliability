# 压缩包清理与代码提炼报告

## 1. 原压缩包问题

原压缩包解压后约 608 MB，包含 3,034 个条目。它把多个研究阶段混在了一起：早期跨范式 benchmark、后续 U-Net 可靠性论文、未完成的 watershed 创新实验、合成测试、调试结果、多个数据集和多轮审稿回复。因此，直接上传 GitHub 会让导师或审稿人难以判断哪些代码真正支撑最终论文。

## 2. 最终论文真正需要的主线

最终论文只需要保留以下链条：

1. BBBC038 固定划分和 100 张测试图像清单。
2. U-Net 训练，随机种子 42、123、2024。
3. 概率阈值 0.50。
4. 删除小于 8 像素的连通组件。
5. 连通组件计数和 GAME 空间误差。
6. IoU 0.50 的一对一实例匹配。
7. 覆盖阈值 0.10 的合并/拆分代理诊断，并检查 0.05、0.20 的敏感性。
8. 预测组件数量分组和高风险组分析。
9. 10,000 次图像级 bootstrap。

新压缩包中的代码、配置、CSV 和图表生成脚本都围绕这条主线组织。

## 3. 删除项目及理由

### 旧论文与审稿文档

删除了所有 `REVIEW_RESPONSE*.md`、`REVIEW_REPORT_CN.md`、`EI_POSITIONING_CN.md`、`CLAIM_LIMITS_CN.md`、`DO_NOT_SUBMIT_BEFORE_RESULTS.md`、`RESULTS_REQUIRED_BEFORE_SUBMISSION.md`、旧 `README_V10...`、`DATA_UPDATE_GUIDE_v15.md`、旧修订日志等。

理由：这些是写作过程记录，不是复现实验所需文件，而且不少内容已经被最终论文取代。

### 未进入最终论文的实验分支

删除了 MCNN、CSRNet、BBBC039、NuInsSeg、Cellpose/StarDist、synthetic sanity check、pilot validation、外部预测导入、跨范式统计比较等代码和结果。

理由：最终论文只报告 BBBC038 上的 U-Net 可靠性诊断。保留这些内容会造成“论文是否还比较了其他模型”的误解。

### watershed 消融分支

删除了 `code/run_postprocessing_ablation.py`、`train_unet_multiseed.py` 的旧脚手架、watershed 流程图和相关说明。

理由：最终论文讨论中只把 watershed 作为未来工作，没有报告其数值结果。把未报告实验放在核心仓库中容易被当成论文证据。

### 重复和过时代码

删除了 `src/make_core_figures_v7.py`、`v8.py`、`v9.py`、v15 图片、多个重复 README、`code/metrics.py` 等。

特别需要注意：旧 `code/metrics.py` 的默认匹配 IoU 是 0.10，而且漏检/误检和 merge 事件的定义与最终论文使用的 `src/metrics.py` 不一致。新仓库只保留一套统一定义。

### 重复结果和调试输出

删除了：

- `results/debug/`
- 未严格设置随机种子的 `results/empirical_runs/`
- MCNN/CSRNet 模型权重和结果
- 重复 U-Net 权重
- `.pyc` 和 `__pycache__`
- demo reference 和结果模板

只保留严格种子版本的 U-Net 逐图 CSV 和训练历史。

### 大文件

删除了原始数据压缩包、转换后图片/掩膜/密度图、模型权重。

理由：这些文件不是“核心代码”，而且会使普通 GitHub 仓库快速膨胀。固定划分清单已经保留，原始数据可从官方来源下载，模型权重可单独发布。

## 4. 新仓库新增的改进

- 统一的 `configs/paper.yaml`。
- 规范的 Python 包结构和 `pyproject.toml`。
- 经过整理的中英文 README。
- 固定 BBBC038 split manifest。
- 可选保存概率图、掩膜、组件标签和 overlap-edge 日志。
- 专门的 proxy 阈值敏感性重算脚本。
- 单元测试。
- `.gitignore` 防止误上传数据和权重。
- GitHub 上传说明。

## 5. 仍需用户决定的事项

- 开源许可证：当前没有自动添加 MIT 或 Apache-2.0，以免未经授权替作者决定知识产权许可。
- 论文最终 DOI、会议名称和正式出版信息：发布后再补充到 `CITATION.cff` 和 README。
- 是否公开模型权重：建议单独放 GitHub Release 或 Git LFS。
