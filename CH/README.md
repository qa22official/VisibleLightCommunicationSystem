# 汉字编解码说明（2048）

这个模块用于把二进制文件和汉字文本互相转换，采用 2048 字字典映射。

## 相关文件

- 脚本：`data_codec/hanzi_codec.py`
- 数据库：`dictionary.db`
- 字典表：`hanzi_map`

## 常用命令

第一次使用先初始化字典：

```powershell
python data_codec\hanzi_codec.py init-dict --db dictionary.db --source example\CH_original.txt
```

把二进制文件编码为汉字 txt：

```powershell
python data_codec\hanzi_codec.py encode --db dictionary.db --input ECC_encode\test\e1.bin --output ECC_encode\output\e1.hanzi.txt
```

把汉字 txt 解码回二进制文件：

```powershell
python data_codec\hanzi_codec.py decode --db dictionary.db --input ECC_encode\output\e1.hanzi.txt --output ECC_encode\output\e1.roundtrip.bin
```

## 编码结果 txt 格式

```text
VLCS-HANZI-2048
bytes=<原始字节数>
crc32=<原始数据CRC32，16进制>
---PAYLOAD---
<汉字载荷内容，多行>
```

## 说明

- 采用 2048 字字典，每个汉字承载 11 bit 数据。
- 解码时会校验 `crc32`，用于检测内容是否损坏。
- 如果 `hanzi_map` 表里已有数据，再执行初始化需要加 `--force` 才会覆盖。
