======
pre-dl
======
implement of <Deep Learning for Precipitation Estimation from Satellite and Rain Gauges Measurements>

.. image:: https://img.shields.io/pypi/v/pre_dl.svg
        :target: https://pypi.python.org/pypi/pre_dl

.. image:: https://img.shields.io/travis/skylight-hm/pre_dl.svg
        :target: https://travis-ci.com/skylight-hm/pre_dl

.. image:: https://readthedocs.org/projects/pre-dl/badge/?version=latest
        :target: https://pre-dl.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

仓储信息
--------
* Python version: python3.7.4
* Operating System: Linux/Windows

### 描述

描述程序的基本架构，约定开发规范。

1.  文件夹结构

.. code-block:: bash
        ├─.github      # github工作流配置文件
        ├─assets       # 数学模型存放位置
        ├─data         # 辅助数据存放位置
        ├─docs          # log.md 是日志文件
        ├─scripts      # 脚本存放位置
        ├─pre_dl       # 库源码
        │  ├─datasets   # 数据集抽象组件
        │  ├─evalator  # 精度评价组件
        │  └─model     # 模型组件
        └─tests         #  单元测试
        └─tmp       #  单元测试时生成的临时文件

2.   开发规范

代码静态检查工具 flake8  
代码整理工具 yapf  
提交前执行单元测试  

