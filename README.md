# Graduation-project
本科毕设_世界模型

## 如何关联本地文件夹与远程仓库

### 方法一：克隆已有仓库

如果你想从远程仓库开始工作，可以直接克隆仓库到本地：

```bash
# 克隆仓库到本地
git clone https://github.com/ai-hua-xia/Graduation-project.git

# 进入项目目录
cd Graduation-project
```

### 方法二：关联现有的本地文件夹

如果你已经有本地代码，想要关联到这个远程仓库：

```bash
# 进入你的本地项目目录
cd /path/to/your/local/project

# 初始化 Git 仓库（如果还没有初始化）
git init

# 添加远程仓库地址
git remote add origin https://github.com/ai-hua-xia/Graduation-project.git

# 验证远程仓库已添加成功
git remote -v
```

## 如何推送本地代码到仓库

### 首次推送

```bash
# 查看当前文件状态
git status

# 添加所有文件到暂存区
git add .

# 或者添加特定文件
# git add filename

# 提交更改
git commit -m "提交说明信息"

# 拉取远程仓库的最新内容（首次推送时建议执行）
git pull origin main --allow-unrelated-histories

# 推送到远程仓库的 main 分支
git push -u origin main
```

### 日常推送

完成首次推送后，日常的代码推送流程：

```bash
# 1. 查看修改的文件
git status

# 2. 添加修改的文件到暂存区
git add .

# 3. 提交更改
git commit -m "描述你的修改内容"

# 4. 推送到远程仓库
git push
```

### 常用 Git 命令

```bash
# 查看提交历史
git log

# 查看远程仓库信息
git remote -v

# 拉取远程最新代码
git pull

# 查看文件修改内容
git diff

# 撤销工作区的修改
git restore filename

# 查看分支
git branch

# 创建并切换到新分支
git switch -c new-branch-name
```

## 注意事项

- 在推送代码前，建议先执行 `git pull` 拉取远程最新代码，避免冲突
- 提交信息（commit message）应该清晰描述本次修改的内容
- 敏感信息（如密码、密钥等）不要提交到仓库中
- 大文件或编译生成的文件建议添加到 `.gitignore` 中
