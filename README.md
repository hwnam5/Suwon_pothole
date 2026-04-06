# 수원시 포트홀 탐지 (Suwon Pothole Detection)

**ROADNet** — Road Observation and Anomaly Detection
**기간:** 2025.02 ~ 2025.06 | **팀원:** 남현원, 공종혁, 한동엽 / 소프트웨어학과 | **지도교수:** 소재현 (교통시스템공학과)

---

## 개요 (Overview)

수원시의 포트홀 발생 수는 꾸준히 증가하고 있으며, 기존 탐지 시스템은 그림자와 포트홀을 구분하지 못하는 문제가 있습니다. 본 프로젝트는 실시간 알림 전달을 통해 수원시가 조기 대응할 수 있도록 돕는 것을 목표로 합니다.

기존 YOLO 기반 Object Detection 방식의 한계를 극복하기 위해 **픽셀 단위 Segmentation 기반의 3단계 파이프라인**을 제안합니다.

---

## 기존 방식의 문제점 (Problem Statement)

| 문제 | 내용 |
|------|------|
| 표현 한계 | Bounding box로는 균열 등 비정형 이상의 정확한 형태·면적 표현 불가 |
| 미세 객체 소실 | Downsampling 과정에서 작은 객체의 feature 손실 |
| 실시간성 한계 | 고해상도 이미지 필요 시 연산량 증가, 해상도 낮추면 탐지 성능 저하 |
| 환경 민감성 | 이상 영역이 흐려지면 detection 실패율 증가 |

---

## 전체 파이프라인 (Overall Pipeline)

```
[차량 주행 영상]
      ↓
[1단계] 1차 분류기 (High-recall Binary Classifier)
      - 포트홀 가능성이 있는 프레임만 필터링 (On-device)
      ↓
[2단계] Segmentation (DeepLabV3)
      - 픽셀 단위로 도로 파손 영역 마스킹
      ↓
[2.5단계] Perspective Transformation (Homography)
      - 원근 왜곡 제거 → 위에서 내려다본 시점으로 변환
      ↓
[3단계] 위험도 분류기 (CBAM + ResNet50)
      - 파손 유형 및 위험도 (0~3단계) 분류
      ↓
[서버] 결과 파일 정리 → 알림 발송 → 보수 조치
```

**Apply 시스템:**
- 차량 주행 중 1차 분류기가 포트홀 의심 프레임만 저장 (통신 비용 절감)
- 서버에서 Main Pipeline을 통해 정밀 탐지 및 위험도 분류
- 결과를 수원시에 알림 전달

---

## 세부 기술 (Technical Details)

### 1. 1차 분류기 — High-recall Binary Classifier

On-device 처리를 위한 경량 모델로, 탐지 누락(FN)을 최소화하는 것이 목표입니다.

**모델:** Pretrained ResNet18
- Transformer 대비 가볍고 inference 속도가 빠름
- 사전학습 모델을 활용하여 소량의 데이터 문제 해결

**핵심 기법:**

**a. Threshold 낮추기**
- 임계값을 낮춰 포트홀일 가능성이 조금이라도 있으면 포트홀로 예측
- FN 감소 목적

**b. Asymmetric Focal Loss (비대칭 집중 손실 함수)**

$$\mathcal{L} = -\sum_{i=1}^{N}\left[y_i \cdot (1-p_i)^{\gamma_{pos}} \cdot \log(p_i) + (1-y_i) \cdot p_i^{\gamma_{neg}} \cdot \log(1-p_i)\right]$$

| 목표 | 설정 |
|------|------|
| FN 줄이기 (positive 더 중요) | γ_pos ↓ |
| FP 허용 (negative 덜 중요) | γ_neg ↑ |

**결과:**

| Loss Function | Accuracy | F1 | Recall |
|---|---|---|---|
| Binary Cross Entropy | 0.9325 | 0.942 | 0.951 |
| γ_pos=0, γ_neg=2 | 0.9327 | 0.944 | 0.959 |
| γ_pos=0, γ_neg=4 | 0.9331 | 0.944 | 0.964 |
| **γ_pos=0, γ_neg=6** | 0.9098 | 0.927 | **0.978** |

관련 코드: [1st_BC/](1st_BC/)

---

### 2. Segmentation — DeepLabV3 (ResNet101 pretrained)

**왜 Detection 대신 Segmentation인가?**

- Pixel 단위 → 실제 pothole의 정확한 형태·면적 반영 가능
- 크기·형태 기반 위험도 분류 가능 (bbox 내부 구조 차이 반영 불가)
- Pothole은 경계가 불명확하고 texture 대비가 낮아 pixel-level 학습이 적합
- 주행 후 로컬 처리 → 속도보다 정확도 우선

**핵심 모듈: ASPP (Atrous Spatial Pyramid Pooling)**

픽셀 주변을 다양한 scale에서 동시에 파악합니다.

| Branch | 설명 |
|--------|------|
| 1×1 conv | Local 정보 |
| 3×3 atrous conv (rate=6) | 중간 범위 context |
| 3×3 atrous conv (rate=12) | 넓은 범위 context |
| 3×3 atrous conv (rate=18) | 더 넓은 scale 처리 |
| Global Average Pooling | 전체 이미지 context |

5개 결과를 `concat → 1×1 conv → 출력`

관련 코드: [2nd_Seg/](2nd_Seg/)

---

### 3. Perspective Transformation (원근 변환)

카메라 시점의 원근 왜곡을 제거하여, 위에서 내려다본 시점(Bird's Eye View)으로 변환합니다.

- 추후 위험도 분류 모델 학습 데이터의 정합성 향상
- OpenCV `cv2.findHomography` 활용

```python
# 도로 ROI 4점 → 직사각형으로 매핑
src_pts = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]  # 원본 도로 영역
dst_pts = [(0,H),   (W,H),   (W,0),   (0,0)    ]  # 정규화된 좌표

H, _ = cv2.findHomography(src_pts, dst_pts)
warped = cv2.warpPerspective(image, H, (W, H))
```

관련 코드: [2_5_Warp/](2_5_Warp/)

---

### 4. 위험도 분류기 — CBAM + ResNet50 (pretrained)

DeepLabV3 Segmentation 결과에서 관심 영역(ROI)을 추출한 뒤, 도로 균열(Crack)인지 포트홀(Pothole)인지 구분하고 포트홀의 경우 위험도를 함께 분류하는 **4-class 분류**를 수행합니다.

**분류 클래스 및 위험도 등급:**

| 클래스 (label) | 파손 유형 | 설명 |
|---|---|---|
| 3 | 포트홀-상 | 포트홀 (픽셀 수 많음, 심각) |
| 2 | 포트홀-중 | 포트홀 (중간 크기) |
| 1 | 포트홀-하 | 포트홀 (픽셀 수 적음, 경미) |
| 0 | 도로 균열 (Crack) | 포트홀 이전 단계의 균열 |

> 포트홀과 균열(Crack)은 모두 도로의 특징을 공유하므로, Segmentation 마스크의 **픽셀 개수**를 기준으로 포트홀 등급(1~3)을 부여하고, 균열은 label 0으로 별도 분류합니다.

**CBAM (Convolutional Block Attention Module)**

기존 ResNet50의 각 Block 출력에 CBAM을 적용하여 어떤 채널과 공간 위치에 집중할지 학습합니다.

- **Channel Attention Module:** C×H×W → C×1×1 (어떤 채널에 집중할지)
  - MaxPool + AvgPool → Shared MLP → Sigmoid
- **Spatial Attention Module:** C×H×W → 1×H×W (어느 픽셀에 집중할지)
  - [MaxPool, AvgPool] concat → Conv → Sigmoid

**학습 데이터 구성:**

1. DeepLabV3로 이미지 Segmentation
2. 배경은 검은색으로 처리, 도로 균열 및 파손 영역은 흰색으로 처리
3. 픽셀 개수에 따라 레이블링 (1~3: 포트홀 등급, 0: 도로 균열)

---

### 5. 추후 성능 개선 시도

정상도로 / Pothole / Crack (label: 0, 1, 2)의 서로 겹치는 특징을 최대한 분리하기 위해 상호 분리 기법 적용된 loss 함수를 실험했습니다.

**MINE Loss (Mutual Information Neural Estimation)**

$$\mathcal{L} = -\left(\mathbb{E}[T(x,z)] - \log\left(\mathbb{E}[e^{T(x,z')}]\right)\right)$$

**Contrastive Loss**
- Pixel-wise label-based contrastive loss
- Projection head를 통해 같은 클래스 pixel은 유사하게, 다른 클래스는 멀어지도록 학습

관련 코드: [2nd_Seg/mine_loss.py](2nd_Seg/mine_loss.py), [2nd_Seg/contrastive_loss.py](2nd_Seg/contrastive_loss.py)

---

## 데이터셋 (Dataset)

- **AI-Hub** 도로 파손 데이터
- **RDD2022** (Road Damage Dataset 2022)
- 수원시 실제 도로 촬영 데이터

관련 코드: [data/](data/)

---

## 프로젝트 구조 (Project Structure)

```
Suwon_pothole/
├── 1st_BC/                          # 1차 분류기 (High-recall Binary Classifier)
│   ├── 1st_BC.ipynb                 # 메인 학습 노트북
│   ├── BCELoss.ipynb                # BCE Loss 실험
│   ├── AsymmetricLoss_gp0_gn2.ipynb # ASL γ_neg=2
│   ├── AsymmetricLoss_gp0_gn4.ipynb # ASL γ_neg=4
│   ├── AsymmetricLoss_gp0_gn6.ipynb # ASL γ_neg=6
│   ├── ASL_0_2.pth / ASL_0_4.pth / ASL_0_6.pth  # 학습된 모델 가중치
│   └── Pothole_BinaryClassification_Pretrained_ResNet18.ipynb
│
├── 2nd_Seg/                         # Segmentation (DeepLabV3)
│   ├── DeepLabv3_train_val_final.ipynb   # 최종 학습
│   ├── DeepLabv3_mine_loss.ipynb         # MINE Loss 실험
│   ├── DeepLabv3_contrastive_loss.ipynb  # Contrastive Loss 실험
│   ├── mine_loss.py                      # MINE Loss 구현
│   └── contrastive_loss.py               # Contrastive Loss 구현
│
├── 2_5_Warp/                        # Perspective Transformation (Homography)
│   ├── WarpROI.py                   # 원근 변환 구현
│   ├── ROI_warp_mask.ipynb          # ROI 설정 및 마스킹
│   └── DeepLab_WarpROI_Stream.ipynb # DeepLab + Warp 연동
│
├── FFC/                             # Fast Fourier Convolution (실험)
│   ├── ffc.py
│   └── ffc_resnet.py
│
└── data/                            # 데이터 전처리
    ├── data2pkl.py / data2pkl.ipynb # 데이터 → pkl 변환
    ├── get_pkl.ipynb
    ├── get_real_data.ipynb          # 실제 수원시 데이터 수집
    └── real_data/
        └── real_data.json
```

---

## 환경 설정 (Requirements)

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib
pip install Pillow
```

---

## 참고 문헌 (References)

- He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
- Chen, L.C. et al. "Rethinking Atrous Convolution for Semantic Image Segmentation." arXiv 2017. (DeepLabV3)
- Woo, S. et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
- Ridnik, T. et al. "Asymmetric Loss For Multi-Label Classification." ICCV 2021.
- Belghazi, M.I. et al. "MINE: Mutual Information Neural Estimation." ICML 2018.
- Khosla, P. et al. "Supervised Contrastive Learning." NeurIPS 2020.
