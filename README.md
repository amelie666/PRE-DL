# pre-dl

implement of <Deep Learning for Precipitation Estimation from Satellite and Rain Gauges Measurements>

仓储信息
--------
* Python version: python3.7.4
* Operating System: Linux/Windows

## 描述

描述程序的基本架构，约定开发规范。

1.  文件夹结构

```bash
tree -d -I __*
├─.github      # github工作流配置文件
├── assets  # 数学模型存放位置
├── data    # 辅助数据存放位置
├── docs    # log.md 是日志文件
├── example  # 模型构建代码
│   ├── PRECNN 
│   ├── PRENet
│   └── QPEML
├── pre_dl    # 库源码
│   ├── datasets  # 数据集抽象组件
│   ├── evaluator  # 数据集抽象组件
│   └── model  # 模型组件
├── scripts  # 脚本存放位置
├── tests   #  单元测试
└── tmp     #  单元测试时生成的临时文件
```
2.   开发规范

代码静态检查工具 flake8  
代码整理工具 yapf  
提交前执行单元测试    

