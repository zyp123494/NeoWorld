#!/bin/bash
set -e

echo "=========================================="
echo "Git 结构配置脚本"
echo "=========================================="
echo ""

cd /home/vivo/桌面/ICLR/World13_opensource

# 1. 处理 splat/ - 移除 .git，作为普通目录
echo "步骤 1: 处理 splat/ 目录..."
if [ -d "splat/.git" ]; then
    echo "  - 备份 splat/.git 到 splat/.git.backup"
    mv splat/.git splat/.git.backup
    echo "  ✓ splat/ 现在将作为普通目录提交"
else
    echo "  ✓ splat/ 已经是普通目录"
fi

# 2. 处理 Amodal3R_align/ - 移除 .git，作为普通目录
echo ""
echo "步骤 2: 处理 Amodal3R_align/ 目录..."
if [ -d "Amodal3R_align/.git" ]; then
    echo "  - 备份 Amodal3R_align/.git 到 Amodal3R_align/.git.backup"
    mv Amodal3R_align/.git Amodal3R_align/.git.backup
    echo "  ✓ Amodal3R_align/ 现在将作为普通目录提交"
else
    echo "  ✓ Amodal3R_align/ 已经是普通目录"
fi

# 3. 处理 ObjectClear/ - 作为 submodule
echo ""
echo "步骤 3: 处理 ObjectClear/ 目录..."

# 检查 ObjectClear 的原始仓库地址
if [ -d "ObjectClear/.git" ]; then
    cd ObjectClear
    OBJECTCLEAR_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
    cd ..
    
    if [ -n "$OBJECTCLEAR_URL" ]; then
        echo "  - 找到原始仓库: $OBJECTCLEAR_URL"
        echo ""
        echo "  要将 ObjectClear 配置为 submodule，需要："
        echo "    1. 先删除 ObjectClear 目录"
        echo "    2. 然后添加为 submodule"
        echo ""
        echo "  是否继续？这会暂时删除 ObjectClear/ (y/n)"
        read -r response
        
        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
            # 备份 ObjectClear 目录
            echo "  - 备份 ObjectClear 到 ObjectClear.backup"
            cp -r ObjectClear ObjectClear.backup
            
            # 删除 ObjectClear
            echo "  - 删除 ObjectClear 目录"
            rm -rf ObjectClear
            
            # 添加为 submodule
            echo "  - 添加为 submodule"
            git submodule add $OBJECTCLEAR_URL ObjectClear
            
            echo "  ✓ ObjectClear 已配置为 submodule"
            echo "  注意：如果你在 ObjectClear 中有未提交的修改，请检查 ObjectClear.backup"
        else
            echo "  跳过 ObjectClear submodule 配置"
        fi
    else
        echo "  ⚠ 无法获取 ObjectClear 的原始仓库地址"
        echo "  请手动配置，或将其作为普通目录提交"
    fi
else
    echo "  ✓ ObjectClear/ 已经是普通目录"
fi

echo ""
echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "目录状态："
echo "  - splat/: 普通目录（包含修改）"
echo "  - Amodal3R_align/: 普通目录（包含修改）"
echo "  - ObjectClear/: $([ -f .gitmodules ] && grep -q ObjectClear .gitmodules && echo 'submodule' || echo '普通目录')"
echo ""
echo "备份文件位置："
echo "  - splat/.git.backup"
echo "  - Amodal3R_align/.git.backup"
[ -d "ObjectClear.backup" ] && echo "  - ObjectClear.backup"

