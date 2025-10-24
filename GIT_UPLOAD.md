# Git 上传指南 / Git Upload Guide

## 重要提示 / Important Notes

在上传前，请确保：
1. ✅ 已创建 `.gitignore` 文件
2. ✅ 已删除所有敏感信息（API keys 等）
3. ✅ 大文件已被排除（模型权重、生成结果等）

## 上传步骤 / Upload Steps

### 1. 初始化 Git 仓库

```bash
cd /home/vivo/桌面/ICLR/World13_opensource
git init
```

### 2. 配置 Git 用户信息（如果还未配置）

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. 添加所有文件

```bash
git add .
```

### 4. 查看将要提交的文件（确认没有敏感信息）

```bash
git status
```

**重要检查项：**
- ❌ 不应包含：`output/` 目录下的大文件
- ❌ 不应包含：`*.pth`, `*.pt` 等模型文件
- ❌ 不应包含：API keys（已在 run.sh 等文件中）
- ✅ 应该包含：源代码文件（`.py`）
- ✅ 应该包含：配置文件（`.yaml`, `.sh`）
- ✅ 应该包含：文档文件（`.md`）

### 5. 提交更改

```bash
git commit -m "Initial commit: NeoWorld - Interactive 3D scene generation with physics and animation"
```

### 6. 关联远程仓库

```bash
git remote add origin git@github.com:zyp123494/NeoWorld.git
```

### 7. 查看远程仓库（确认配置正确）

```bash
git remote -v
```

### 8. 推送到 GitHub

```bash
# 首次推送
git push -u origin main

# 如果远程仓库已有内容，可能需要先拉取：
# git pull origin main --allow-unrelated-histories
# git push -u origin main
```

## 后续更新 / Future Updates

当你修改代码后，使用以下命令更新：

```bash
# 1. 查看修改
git status

# 2. 添加修改的文件
git add <文件名>
# 或添加所有修改
git add .

# 3. 提交
git commit -m "描述你的修改"

# 4. 推送
git push
```

## 常见问题 / Troubleshooting

### 问题 1: 文件太大无法推送

如果某些文件太大，编辑 `.gitignore` 添加：
```
output/
*.pth
*.pt
*.ckpt
```

然后移除已添加的大文件：
```bash
git rm --cached -r output/
git commit -m "Remove large files"
```

### 问题 2: 认证失败

确保你已经设置了 SSH key：
```bash
# 生成 SSH key（如果还没有）
ssh-keygen -t ed25519 -C "your.email@example.com"

# 将公钥添加到 GitHub
cat ~/.ssh/id_ed25519.pub
# 复制输出，然后到 GitHub Settings > SSH Keys 添加
```

### 问题 3: 分支名称问题

如果你的默认分支是 `master` 而不是 `main`：
```bash
# 重命名分支
git branch -M main
git push -u origin main
```

## Git LFS（可选，用于大文件）

如果你需要上传一些大文件（如预训练模型）：

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件类型
git lfs track "*.pth"
git lfs track "*.pt"

# 添加 .gitattributes
git add .gitattributes

# 正常提交和推送
git add .
git commit -m "Add large files with LFS"
git push
```

## 检查清单 / Checklist

上传前请确认：
- [ ] `.gitignore` 已创建并配置
- [ ] 敏感信息已删除（API keys）
- [ ] README.md 已更新
- [ ] 代码已测试运行正常
- [ ] commit 信息清晰明了
- [ ] 远程仓库地址正确

