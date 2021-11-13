# ♻재활용 품목 분류를 위한 Semantic Segmentation Report♻

# 목차

- [팀소개](#팀소개)
- [대회 개요](#대회-개요)
- [개발환경 및 활용 장비](#개발환경-및-활용-장비)
- [문제 정의](#문제-정의)
- [실험 내용](#문제에-대한-실험)
- [Modeling 및 Ensemble](#Modeling-및-Ensemble)
- [회고](#회고)

# 팀소개


<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Seoheesu1">
        <img src="https://avatars.githubusercontent.com/u/63832160?v=4" width="100px;" alt=""/>
        <br />
        <sub>서희수</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/WonsangHwang">
        <img src="https://avatars.githubusercontent.com/u/49892621?v=4" width="100px;" alt=""/>
        <br />
        <sub>황원상</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sala0320">
        <img src="https://avatars.githubusercontent.com/u/49435163?v=4" width="100px;" alt=""/>
        <br />
        <sub>조혜원</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hongsusoo">
        <img src="https://avatars.githubusercontent.com/u/77658029?v=4" width="100px;" alt=""/>
        <br />
        <sub>홍요한</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Junhyuk93">
        <img src="https://avatars.githubusercontent.com/u/61610411?v=4" width="100px;" alt=""/>
        <br />
        <sub>박준혁</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hanlyang0522">
        <img src="https://avatars.githubusercontent.com/u/67934041?v=4" width="100px;" alt=""/>
        <br />
        <sub>박범수</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GunwooHan">
        <img src="https://avatars.githubusercontent.com/u/76226252?v=4" width="100px;" alt=""/>
        <br />
        <sub>한건우</sub>
      </a>
    </td>
  </tr>
  <tr>
    </td>
  </tr>
</table>
<br>  

# 대회 개요

![](https://i.imgur.com/PnOdQ0L.png)

재활용 쓰레기 사진에 대하여 일반 쓰레기, 플라스틱, 종이, 유리 등의 10 종류의 재활용 품목으로 semantic segmentation을 수행 🌎


![](https://s3-us-west-2.amazonaws.com/aistages-dev-server-public/app/Users/00000274/files/2c237ded-2980-4cc8-8dad-fa46ead2e2a6..png)

- **Input :** 쓰레기 객체가 담긴 이미지. Segmentation annotation은 COCO format.
  - 11 class : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing


- **Output** : bbox 좌표, 카테고리, score 값을  submission 양식에 맞게 csv 파일을 만들어 제출.

## 평가 방법

Test set의 **mIoU**로 평가
- Semantic Segmentation에서 사용하는 대표적인 성능 측정 방법
- Ground Truth pixel과 Prediction Pixel간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)의 평균으로 계산.
- model로부터 예측된 mask의 size는 512 x 512 지만, 대회의 원활한 운영을 위해 output을 일괄적으로 256 x 256 으로 변경해 score 반영
- mIoU 수식 : 모든 Class에 대해서 IoU값의 평균을 구함
    ![image](https://user-images.githubusercontent.com/77658029/141066763-279352d5-36a4-42a2-8b8f-a34cc5609d5f.png)
    
- Example of mIoU  
    ![image](https://user-images.githubusercontent.com/77658029/141066898-e32827ad-1495-4f42-bbd3-a159bd72a605.png)



 
        

# 개발환경 및 활용 장비

![](https://i.imgur.com/rh7Rweg.png)

<table class="c25"><tbody><tr class="c83"><td class="c82" colspan="1" rowspan="1"><p class="c19"><span class="c43 c38"><strong>Development Environment</strong></span></p></td><td class="c39" colspan="1" rowspan="1"><p class="c19"><span class="c43 c38"><strong>Reference Repository</strong></span></p></td><td class="c32" colspan="3" rowspan="1"><p class="c19"><span class="c43 c38"><strong>Virtual Environment Library</strong></span></p></td></tr><tr class="c89"><td class="c99" colspan="1" rowspan="2"><p class="c19"><span class="c7">Language : Python 3.7.11</span></p><p class="c19"><span class="c7">OS : Ubuntu 18.04.5 LTS</span></p><p class="c19"><span class="c7">GPU : V100</span></p></td><td class="c36" colspan="1" rowspan="1"><p class="c19"><span class="c88 c20"><a class="c55" href="https://www.google.com/url?q=https://github.com/open-mmlab/mmsegmentation&amp;sa=D&amp;source=editors&amp;ust=1636531716293000&amp;usg=AOvVaw1bRHSv3HB19VoCHryLfo9A">MMSegmentation</a></span></p></td><td class="c76" colspan="2" rowspan="1"><p class="c21"><span class="c7">matplotlib 3.4.3</span></p><p class="c21"><span class="c7">numpy 1.21.2</span></p><p class="c21"><span class="c7">pillow 8.3.2</span></p><p class="c21"><span class="c7">opencv-python 4.5.3.56</span></p></td><td class="c97" colspan="1" rowspan="1"><p class="c16"><span class="c7">mmcv-full 1.3.16</span></p><p class="c16"><span class="c20">pytorch</span><span class="c7">&nbsp;1.7.0</span></p><p class="c16"><span class="c7">torchvision 0.8.0</span></p><p class="c16"><span class="c20">mmsegmentation</span><span class="c7">&nbsp;0.18.0</span></p></td></tr><tr class="c86"><td class="c36" colspan="1" rowspan="1"><p class="c19"><span class="c20 c88"><a class="c55" href="https://www.google.com/url?q=https://github.com/qubvel/segmentation_models.pytorch&amp;sa=D&amp;source=editors&amp;ust=1636531716297000&amp;usg=AOvVaw1smUvmqx-TGrelwNp_nIBp">Segmentation Models</a></span></p></td><td class="c76" colspan="2" rowspan="1"><p class="c21"><span class="c7">matplotlib 3.4.3</span></p><p class="c21"><span class="c7">numpy 1.21.2</span></p><p class="c21"><span class="c7">pillow 8.4.0</span></p><p class="c21"><span class="c7">opencv-python 4.5.3.58</span></p></td><td class="c97" colspan="1" rowspan="1"><p class="c16"><span class="c20">pytorch</span><span class="c7">&nbsp;1.7.0</span></p><p class="c16"><span class="c7">torchvision 0.8.0</span></p><p class="c16"><span class="c7">timm 0.4.12</span></p><p class="c16"><span class="c7">smp 0.2.0</span></p></td></tr></tbody></table>


# 문제 정의

- __시각화__

![](https://i.imgur.com/4xEJNeP.png)

- Class별 annotation 수의 불균형이 심하고, 평균 2,314개인 data 갯수에 비해 배터리는 데이터 수가 63개뿐

<p align="center"><img src="https://i.imgur.com/5o6uKA3.png"></p>
<p align="center"><img src="https://i.imgur.com/luGSC26.png"></p>


**① Perceptual distance를 이용한 val dataset 검증**

<p align="center"><img src="https://i.imgur.com/tF9kkrl.png"></p>

- VGG19 Imagenet Pretrained Model을 이용하여 High-level feature map을 추출함
- feature map은 High-level feature을 잘 뽑는다고 알려진 conv4_4 output을 사용함 (CartoonGAN, Perceptual loss 논문 ref)
- feature map 추출 이후 global average pooling을 이용하여 128 차원의 1d vector로 만든 후, autoencoder에 학습시켜 클러스터링 하였음
- High-level feature가 비슷할 경우, 분포하는 class가 유사할 것이라고 생각하여 진행함
- clustering 결과, train, test, valid set이 유사한 분포를 보였음. 대신 train data(녹)의 경우 valid나 test에 비해 좀 더 넓은 분포를 가지고 있고, 몇몇 데이터는 분포에서 많이 멀어져있는 것을 확인함. 해당 데이터들이 노이즈 형태로 학습에 악영향을 미칠 수 있다고 판단하였음 



**② Class Dependency**
- 전단지의 경우 일반 쓰레기와 종이 두 가지로 annotation되어 있음
- 유리와 투명 플라스틱이 매끈한 표면, 투명함 등 이미지상에서 유사한 특징을 보임
- 얇은 물체(노끈이나 줄 등)에 대한 background Error가 아주 높은 경향을 보임

**③ Class Imbalance**
- Figure 1에서 처럼 Class별 bbox 수의 불균형이 심함
- 배터리의 경우, 데이터가 63개로 다른 class에 비해 현저히 적음

**④ Various Dataset Environment**
- Figure 2 와 같이 다양한 환경에서 촬영된 이미지



# 문제에 대한 실험

## 실험 결과

### Issue 및 성능 개선을 위한 시도

**① Data Augmentation** : Class Imbalance 및 Image의 촬영 환경 보완을 위한 다양한 Augmentation 기법 시도
→ Rotate, RandomResizedCrop, MotionBlur, GridDistortion, HueSaturationValue, RandomBrightnessContrast, ImageCompression, Hor/VerFlip

**② Model Selection** : 최적의 모델을 찾기 위해 다양한 모델로 실험

**③ Generalization** : 여러 Augmentation과 Noise를 넣어 시도

**④ Pseudo Labeling** : 학습한 모델로 test 데이터를 inference한 후, 그 결과로 추가 학습

**⑤ CRF(Conditional Random field)** : denseCRF 후처리를 통해 픽셀단위의 정확도 향상 도모

**⑥ Ensemble** : 여러 모델을 Ensemble(soft or hard voting) 함으로서 Robust한 모델 개선 시도

**⑦ YohanMix** : 클래스 불균형 해소를 위해 적은 개수의 클래스의 image를 기존 dataset에 CutMix와 같은 방식으로 이어붙이는 방식.

**⑧ TTA(Test Time Augmentation)** : 학습 때와 다른 input image를 통해 inference 하는 방법 / Multiscale

### 모델 선정

<table class="c79"><tbody><tr class="c60"><td class="c68" colspan="1" rowspan="1"><p class="c9"><span class="c43 c38">Model</span></p></td><td class="c110" colspan="1" rowspan="1"><p class="c9"><span class="c43 c38">설명</span></p></td><td class="c63" colspan="1" rowspan="1"><p class="c9"><span class="c43 c38">mIOU (LB Score)</span></p></td></tr><tr class="c3"><td class="c77" colspan="1" rowspan="1"><p class="c9"><span class="c7">DeepLabV3++ (se-resnext)</span></p></td><td class="c53" colspan="1" rowspan="1"><p class="c64"><span class="c37">CNN에서 semantic info를 더 detail하게 고려하기 위해 만든 모델</span></p></td><td class="c70" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.700</span></p></td></tr><tr class="c3"><td class="c77" colspan="1" rowspan="1"><p class="c9"><span class="c7">UPerNet</span></p><p class="c9"><span class="c7">&nbsp;(Swin)</span></p></td><td class="c53" colspan="1" rowspan="1"><p class="c64"><span class="c37">Parts, Meterial, Scene, Objects, Textures 같이 다양한 visual concept을 파싱한 후, 이 정보들을 통합 이용하여 segmentation을 수행 </span></p></td><td class="c70" colspan="1" rowspan="1"><p class="c9"><span class="c17">0767</span></p></td></tr><tr class="c85"><td class="c77" colspan="1" rowspan="1"><p class="c9"><span class="c7">OCRNet</span></p><p class="c9"><span class="c7">(HRNet)</span></p></td><td class="c53" colspan="1" rowspan="1"><p class="c64"><span class="c37">객체와의 관계를 통해서 각 픽셀의 정보를 유추하는 OCR을 통해 pixel representations을 강화하고, HRNet을 사용해서 중요한 부분을 high resolution 정보를 계속 추가해주는 방식의 모델</span></p></td><td class="c70" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.705</span></p></td></tr></tbody></table>

## 실험 히스토리
<p align="center"><b>Wandb를 활용한 실험 관리</b></p>

![](https://i.imgur.com/BZX0z9H.png)




## 검증 전략
- Stratified validation과 Confusion Matrix를 통해 분류 모델 평가
- class별 mIOU를 validation set으로 확인하여 모델 평가 
- 모든 모델이 General trash, Paper pack, Plastic Class에서
	정확도가 떨어지는 경향성을 보임


<p align="center"><img src="https://i.imgur.com/aIfSPjD.png"></p>



<!-- <div>
<center><img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fe34061b-1c71-4d04-9e68-94c8a462e7d9/confusion_matrix.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211016%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211016T113908Z&X-Amz-Expires=86400&X-Amz-Signature=fe612e73417194c63f47e5655c597bb2c726182b55ac2cc9f89f9ec9a2ac2a71&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22confusion_matrix.png%22"></center>
-->

</div>





# Modeling 및 Ensemble

<table class="c102"><tbody><tr class="c60"><td class="c109" colspan="1" rowspan="1"><p class="c9"><span class="c17">Architecture</span></p></td><td class="c94" colspan="1" rowspan="1"><p class="c58"><span class="c17">Backbone Model</span></p></td><td class="c100" colspan="1" rowspan="1"><p class="c9"><span class="c59 c38 c56 c40">테스트 및 개선 시도</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c9"><span class="c17">mIOU (LB)</span></p></td></tr><tr class="c3"><td class="c15" colspan="1" rowspan="7"><p class="c9"><span class="c7"><br><br>DeepLabv3+</span></p><p class="c9 c33"><span class="c7"></span></p></td><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c20">resnet 152</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c54">0.646</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c20">se_resnet 152</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c54">0.611</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">senet 154</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.671</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">regnety</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.669</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">dpn</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.637</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">efficentnet-b7</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.652</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9 c29"><span class="c20">se_resnext 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">warmup cosine annealing , Augmentation</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.7</span></p></td></tr><tr class="c3"><td class="c15" colspan="1" rowspan="7"><p class="c9"><span class="c7">Upernet</span></p><p class="c9 c33"><span class="c7"></span></p></td><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-B</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">기본</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.755</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-B</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">dense-CRF iter 15회</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.755</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-B</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">TTA (multi-scale)</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.764</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-B</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">K-fold (k=5) hard-voting ensemble</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.762</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-B</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">TTA (flip, multi-scale)</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.766</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-B</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">pseudo labeling, TTA (flip, multi-scale)</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.776</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">Swin-L</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.749</span></p></td></tr><tr class="c24"><td class="c15" colspan="1" rowspan="2"><p class="c9"><span class="c7">OCRNet</span></p><p class="c9 c33"><span class="c7"></span></p></td><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">HRNetV2p-W48</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">기본</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.695</span></p></td></tr><tr class="c3"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">HRNetV2p-W48</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">TTA (multi-scale)</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.705</span></p></td></tr><tr class="c23"><td class="c15" colspan="1" rowspan="3"><p class="c9"><span class="c20">UNet++</span></p></td><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">se_resnext 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c17">0.655</span></p></td></tr><tr class="c31"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">HRNet</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.569</span></p></td></tr><tr class="c31"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">resnet</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c7">0.472</span></p></td></tr><tr class="c35"><td class="c15" colspan="1" rowspan="3"><p class="c9"><span class="c7">FCN</span></p></td><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">resnet 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c20">cutmix</span><span class="c7">&nbsp;미사용</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c20 c91">0.5624</span></p></td></tr><tr class="c35"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">resnet 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c20">cutmix</span><span class="c7">&nbsp;battery 추가</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c20 c78 c91">0.5571</span></p></td></tr><tr class="c35"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">resnet 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c20">cutmix</span><span class="c7">&nbsp;metal, glass, battery, clothing 추가</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c38 c78 c40 c91">0.5688</span></p></td></tr><tr class="c31"><td class="c15" colspan="1" rowspan="2"><p class="c9"><span class="c7">PAN</span></p></td><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">resnet 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c49 c20">0.585</span></p></td></tr><tr class="c35"><td class="c11" colspan="1" rowspan="1"><p class="c9"><span class="c7">se_resnext 101</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c9"><span class="c7">-</span></p></td><td class="c14" colspan="1" rowspan="1"><p class="c9"><span class="c38 c40 c49">0.653</span></p></td></tr></tbody></table>

더 자세한 기록은 [더보기...](https://docs.google.com/spreadsheets/d/1xw11I8pUZY8CGaE0jXeE4KGokvK-zbpxHdKKYsyt_SI/edit?usp=sharing)

## 최종 Ensemble된 모형
<p align="center"><img src="https://i.imgur.com/yAn7tVS.png"></p>
각 model별로 가장 LB score가 좋았던 버전 5개를 Ensemble하여 최종 모델을 생성  



### 시연 결과

<p align="center"><img src=https://i.imgur.com/cfoPCei.png></p>

- 모델 별로 inference결과가 다르게 나와서 ensemble 시도하였더니 성능이 더 좋아진 것으로 보임
- 전체적으로 OCR은 object의 모양과 texture 학습하는 경향이 있고, Swin은 image의 맥락을 학습하는 경향이 있는 것으로 사료됨

                       
                       
## 회고

### 팀내 자체 평가 및 아쉬운 점

- mmsegmentation과 관련된 다양한 라이브러리를 처음 활용함에 어려움이 있어서 내가 원하던 모델을 구현하는데 시간이 오래 걸림. 새롭게 배워간다기 보다 mmsegmentation 라이브러리를 통해 기계적으로 파라미터  튜닝을 하여 성능을 향상시킨다는 느낌이 강해서 아쉬웠음
- test dataset과 유사한 validation dataset을 찾기 어려워 연구가 진행될수록 model을 개선시키기 위한 뚜렷한 판단의 근거가 부족했음
- SMP에서 실험한 augmentation test가 mmseg에서는 적용 불가능하거나, 동일한 효과를 보이지 못해서 아쉬웠음
- Pseudo labeling 이외의 방법들은 모두 public-private LB가 많이 차이나서 robust한 모델을 만들지 못했다는 생각이 들어 아쉬웠음
- 성능평가 및 결과 분석, EDA를 통해 개선점을 찾아나가는 과정이 부족했다는 것이 아쉬움
- 개선 아이디어를 좁은 범위에서만, 이미 알려진 것에서만 찾았던 것 같음. (augmentation, ensemble 등) 재활용 데이터 셋 특징에 맞는 개선 아이디어를 찾지 못했던 것 같음. 딥러닝 범위 밖에서도 생각해보고, 적용가능성을 검토해봤어야 했음
- SOTA 모델이 무엇인지 같은 정보는 검색을 통해 찾아 볼 수 있었지만, 모델을 어떻게 운영할 지에 대한 경험이 부족해, 전체적인 상황을 보는 시각이 좁다는 느낌을 받았음
- 기존 대회와 다른 뭔가 기발한 방법을 적용시켜보고자 했으나, 크게 다르거나 획기적인 방법을 찾아내지 못했음.


### 연구 방향 제시
- Object만 고려하기보다 semantic info를 함께 고려하는 architecture를 사용한다면 더 성능이 높아질 것이라 생각됨
- 전체적으로 General trash, Paper pack, Plastic Class에 대해서 정확도가 떨어졌으며 이에 대한 해결책으로 다양한 augmentation을 추가해보았으나 성능상 큰 효과를 나타내지 못하였고, 이를 해결하기 위한 다른 방안이 있다면 더 좋은 결과를 도출해낼 수 있을 것이라 생각됨.
- 애매한 object는 따로 class를 만들어 검출한 후 기존 class에 합친다면 효과가 올라갈 것으로 예상
- Dense CRF 시도시 예측한 마스크의 형태를 Image에 맞게 조정해주는 역할을 해줬는데, 예측한 Mask의 크기를 키운 후 Dense CRF 시도하면 성능 개선에 도움이 될것이라 생각됨



## 팀원들의 한마디

- 한건우 : benchmark로 사용된 데이터셋의 경향과 우리가 해결하고자 하는 문제의 데이터셋의 경향을 고려해야한다는 것을 알게 되었습니다
- 박준혁 : 
- 홍요한 : 너무 Cutmix 적용에만 몰두하여 다른 시도들을 못했던게 아쉬웠습니다. 다음 대회에서는 계획을 세워 일정에 맞춰 진행해 봐야겠습니다.
- 황원상 : 
- 박범수 : 
- 서희수 : 모델을 바꿔 보는 것 이외의 다른 개선 시도를 해보고자 했으나 적합한 방법을 찾아내기 어려웠던 점이 아쉬웠습니다. 모두 고생 많으셨습니다!!
- 조혜원 : 

# Reference

<p><span style="background-color:#EEEEEE;">네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0<br/>
https://stages.ai/competitions/76/overview/description
</span></p>
