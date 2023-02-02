# UDA - Semantic Segmentation

## Requirements

- PyTorch 1.3 to 1.6

## Dataset

### <MF Dataset>
- 도심지에서 낮과 밤 시간대에 촬영된 데이터 셋으로 쌍을 이루는 RGB 영상과 Thermal 영상을 제공함.
- 픽셀 단위의 시맨틱 라벨 정보를 제공하고 있어 시맨틱 정보 추정 연구에 활용 됨.

- [데이터 셋 다운 홈페이지](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) 이곳에서 [Multi-spectral Semantic Segmentation Dataset (link to Google Drive)](https://drive.google.com/drive/folders/1YtEMiUC8sC0iL9rONNv96n5jWuIsWrVY) 이 링크를 통해서 Dataset을  현재 경로 다운 받아야한다.

- MF Dataset은 RGB 영상과 Thermal 영상을 합쳐서 4채널로 제공은 한다.
- 따라서 두 도메인의 영상을 따로 다루기 편하도록 RGB 영상과 Thermal 영상 따로 저장하는 작업이 필요하다.
- ```Make_split.ipynb```을 이용해 RGB 와 Thermal 를 분리해 저장해야한다. 
- 추가적인 Dataset 관련 셋팅은 [MS-UDA](https://github.com/yeong5366/MS-UDA) 참고

## Dataloader


- 데이터 폴더 구조 :
```
data
├── ir_seg_dataset
│   ├── images
│   │   ├── 00001D.png
│   │   ├── 00003N.png
│   │   ├── 00006N.png
│   │   └── ...
│   ├── labels
│   │   ├── 00001D.png
│   │   ├── 00003N.png
│   │   ├── 00006N.png
│   │   └── ...
│   └── ...
├── models
├── output

```

## Train && Test 

### 학습 및 평가 방식 
- 학습
   ```
    bash scripts/train.sh
   ``` 
- 평가
   ```
    bash scripts/test.sh
   ``` 
