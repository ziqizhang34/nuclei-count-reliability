# 上传 GitHub 操作步骤

## 方法一：使用网页和命令行

### 1. 在 GitHub 新建空仓库

仓库名可使用：

```text
nuclei-count-reliability
```

创建时不要勾选自动添加 README、`.gitignore` 或 License，因为压缩包中已经有 README 和 `.gitignore`，License 需要作者自己选择。

### 2. 解压本压缩包并进入目录

```bash
cd nuclei_count_reliability_github
```

### 3. 初始化并提交

```bash
git init
git branch -M main
git add .
git commit -m "Initial release: U-Net nuclei-count reliability analysis"
```

### 4. 连接远程仓库

把下面地址替换为你的 GitHub 用户名和仓库名：

```bash
git remote add origin https://github.com/YOUR_USERNAME/nuclei-count-reliability.git
git push -u origin main
```

## 方法二：GitHub Desktop

1. 解压本压缩包。
2. 打开 GitHub Desktop。
3. 选择 `File` -> `Add local repository`。
4. 选择解压后的文件夹。
5. 提交说明填写 `Initial release`。
6. 点击 `Publish repository`。

## 不要直接提交的内容

`.gitignore` 已经排除：

- 原始和转换后的图片数据。
- `.pt`、`.pth`、`.ckpt` 模型权重。
- 概率图、组件标签和 overlap 日志。
- Python 缓存和虚拟环境。

需要公开权重时，优先使用 GitHub Release；需要版本化追踪大型权重时，再考虑 Git LFS。

## 上传前最后检查

```bash
pytest
nuclei-analyze --results-root results/per_seed --out-dir results/generated

git status
```

确认 `git status` 中没有出现原始数据、模型权重或个人临时文件，再推送。
