# TTS-MOS-Easy-Survey

[English](./readme_en.md) | 中文

一个单文件的简单易用基于Gradio的 MOS (平均意见得分) 评测系统,适用于TTS（文本转语音），VC（语言转换）等模型评估。

## Demo
![demo](assets/demo.png)

## 功能特点

- 单个代码文件
- 支持若干个 TTS 模型的音频对比评测
- 自动随机化音频播放顺序
- 防止重复提交评测
- 评分结果自动保存为 CSV 格式
- 简洁的网页界面

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```
2. 准备音频文件：
   - 在 `examples/prompt/` 目录下放置参考音频
   - 在 `examples/` 下为每个模型创建子文件夹，放入对应的合成音频，文件夹名称请与模型名称一致
   - 模型文件夹名称将被记录到result.csv中
   - 确保所有模型文件夹中的音频数量与参考音频相同
   - 确保所有文件夹内的音频文件命名顺序，系统会自动排序：1.wav, 2.wav, 3.wav, ...

3. 运行评测系统：
```bash
python app.py
```
4. 在浏览器中访问 `http://localhost:8565` 开始评测

## 评分标准 
1-5分，0.5分一档
| 分数 | 自然度/人声相似度 | 机器音特征 |
|-------|---------------------|-------------|
| 5 优秀 | 完全自然的语音 | 无法察觉机器音特征 |
| 4 良好 | 大部分自然的语音 | 可以察觉但不影响听感 |
| 3 一般 | 自然与不自然程度相当 | 明显可察觉且略有影响 |
| 2 较差 | 大部分不自然的语音 | 令人不适但尚可接受 |
| 1 很差 | 完全不自然的语音 | 非常明显且无法接受 |

## 结果保存

评测结果将保存在 `results/results.csv`

```csv
id,model,MOS1,MOS2,MOS3,...
aasda,fish,1,1.5,3
aasda,30w,2,4.5,3
aasda,100w,2.5,1,3
rer,fish,1,1.5,2
rer,30w,2.5,2,4.5
rer,100w,1.5,3.5,4
```

## Reference
- https://github.com/coqui-ai/TTS/discussions/482#discussioncomment-10772959