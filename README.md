# 运行流程

## 问题描述和参数的输入和存储
### 运行方式 A：命令行直接传描述
```python
python main.py -n demo-ed -d "有一个电力系统...EOF不需要，这里整段是单行或转义的多行"
```

多行可用引号里的 \n，或用三引号在某些 shell 中传递。

### 运行方式 B：从文件读取描述

```python
# desc.txt 是你准备好的自然语言描述（UTF-8）
python main.py -n demo-ed -f ./desc.txt
```

### 运行方式 C：交互式多行输入（你原函数的流程）
直接不传 -d/-f，程序会提示输入。逐行输入后，单独一行输入 EOF 结束。

Linux/macOS 也可用快捷 EOF：输入完成后 Ctrl-D。

Windows 终端可用 Ctrl-Z 然后回车。
```python
python main.py -n demo-ed
Enter the optimization problem description. Type EOF on a new line when done:
这里写第1行
这里写第2行
EOF
```

### 覆盖策略
如果同名问题目录 `problems/<name>/` 已存在，`main.py` 会提示是否覆盖。
想无提示覆盖，加 `--force`：
```python
python main.py -n demo-ed -f ./desc.txt --force
```

### 查看输出
成功后会打印两个文件路径：
```python
problems/<name>/desc.txt：原始描述（UTF-8）。

problems/<name>/state_0_description.json：初始状态（包含 description、空 parameters、meta）。
```