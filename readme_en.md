# TTS-MOS-Easy-Survey

[English](./readme_en.md) | [中文](./readme.md)

A simple single-file MOS (Mean Opinion Score) evaluation system for TTS (Text-to-Speech), VC (Voice Conversion)... model evaluation.

## Features

- Single code file
- Support for comparing multiple TTS models
- Automatic randomization of audio playback order
- Prevention of duplicate submissions
- Results automatically saved in CSV format
- Clean web interface

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Prepare audio files:
   - Place reference audio files in the `examples/prompt/` directory
   - Create subdirectories for each model in `examples/`, place corresponding synthesized audio files, directory names should match model names
   - Model folder names will be recorded in result.csv
   - Ensure all model folders contain the same number of audio files as reference audio
   - Ensure consistent file naming order in all folders, system will sort automatically: 1.wav, 2.wav, 3.wav, ...

3. Run the evaluation system:
```bash
python app.py
```
4. Visit `http://localhost:8565` in your browser to start evaluation

## Scoring Criteria
1-5 points, with 0.5 point intervals
| Score | Naturalness/Human Similarity | Machine Characteristics |
|-------|----------------------------|------------------------|
| 5 Excellent | Completely natural speech | No detectable machine characteristics |
| 4 Good | Mostly natural speech | Detectable but doesn't affect listening experience |
| 3 Fair | Equal natural and unnatural qualities | Clearly detectable with slight impact |
| 2 Poor | Mostly unnatural speech | Uncomfortable but still acceptable |
| 1 Bad | Completely unnatural speech | Very obvious and unacceptable |

## Results Storage

Evaluation results will be saved in `results/results.csv`

```csv
id,model,MOS1,MOS2,MOS3,...
aasda,fish,1,1.5,3
aasda,30w,2,4.5,3
aasda,100w,2.5,1,3
rer,fish,1,1.5,2
rer,30w,2.5,2,4.5
rer,100w,1.5,3.5,4
``` 
