# tf2-speech-recognition-las
Tensorflow 2 Speech Recognition Code (LAS)

<center><img src="./image/pic1.png"></center>


## Developers
* Younglo Lee (yllee@ispl.korea.ac.kr)

## Contents
  * [Contents](#contents)
  * [Speech Recognition](#speechrecognition)
  * [Features](#features)
  * [Prerequisites](#prerequisites)
  * [Examples](#examples)
  * [References](#references)
    
## Speech Recognition
- Tensorflow 2 implementation for korean speech recognition.
- Dataset can be downloaded in http://www.aihub.or.kr/aidata/105/download

## Features
- Listen, Attend and Spell (LAS) speech recognition model
- Location sensitive attention
- Korean syllable tokenization

## Prerequisites
- Python 3.x
- Tensorflow 2.2.0
- Librosa 0.6.3
- etc.

## Examples
- Tensorboard examples (trained on data with short length (<=500))
<center><img src="./image/pic2.png"></center>
- Attention weight examples
<center><img src="./image/pic3.png"></center>

## References and Resources
- https://github.com/Rayhane-mamah/Tacotron-2
- https://github.com/clovaai/ClovaCall
