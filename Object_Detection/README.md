# 🌏재활용 품목 분류를 위한 Object Detection Report🌏

# 목차

- [팀소개](#팀소개)
- [대회 개요](#대회-개요)
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

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

- **Input :** 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리)가 모델의 인풋으로 사용됩니다. bbox annotation은 COCO format으로 제공됩니다. (COCO format에 대한 설명은 학습 데이터 개요를 참고해주세요.)
- **Output** : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다. (submission format에 대한 설명은 평가방법을 참고해주세요

### 평가 방법

Test set의 mAP50(Mean Average Precision)로 평가
- Object Detection에서 사용하는 대표적인 성능 측정 방법
- Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단합니다.
- Example of IoU  
        ![](https://i.imgur.com/lb9BsAG.jpg)

- metric  
        ![](https://i.imgur.com/7WwZGOb.png)

        
- Example of mAP50  
        ![](https://i.imgur.com/387jIEL.png)

        
    - **Orange**
        - TP = 1, FP = 1
        - 총 2개의 Orange 박스 중 하단의 박스 1개는 객체를 잘 detection하였습니다. (TP) 
            상단의 박스 1개는 Blue category에 해당하는 객체를 Orange category로 예측하였기 때문에 잘못된 detection입니다. (FP)
            
        ![](https://i.imgur.com/Wk7hrji.png)

        
    - **Blue**
        - TP = 2, FP = 1  
        - 총 3개의 Blue 박스 중 두 개의 박스는 객체를 잘 detection하였습니다. (TP) 
            우측 하단의 박스는 객체 위치를 정확히 detection하지 못했습니다. (FP)
        
        ![](https://i.imgur.com/H2ycuEq.png)
        
        - **mAP**
            
            모든 이미지의 각 클래스별 AP 계산 후, 평균내어 최종 점수가 구해집니다.
            
        
        ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/005d7efe-0301-489f-9146-ef4ad56945a3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211016%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211016T113948Z&X-Amz-Expires=86400&X-Amz-Signature=daec222beca2631f8c96f355adc1cdfa8ade2122eb76941a8de4c94909b70365&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
        





# 문제 정의

- __시각화__

![images](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b0ff425c-f802-4188-a4df-41b1befbd3eb/BoxesPerCategory.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211016%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211016T114011Z&X-Amz-Expires=86400&X-Amz-Signature=ca7a2ae161882c480c242d4ee9bfa9b5649152beee74cf07809810929b373c8e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22BoxesPerCategory.png%22)
위 이미지에서 보이는 바와 같이 Class별 bbox 수의 불균형이 심하고, 배터리의 경우 데이터 수가 159개(평균 2,314 개)뿐입니다.

<p align="center"><img src="https://i.imgur.com/DFKDvXF.png"></p>
<p align="center"><img src="https://i.imgur.com/H27mkEx.png"></p>


① Class Dependency
- 전단지의 경우 일반 쓰레기와 종이 두 가지로 annotation되어 있음
- 유리와 투명 플라스틱이 매끈한 표면, 투명함 등 이미지상에서 유사한 특징을 보임
- 얇은 물체(노끈이나 줄 등)에 대한 background Error가 아주 높은 경향을 보임

② Class Imbalance
- Figure 1에서 처럼 Class별 bbox 수의 불균형이 심함
- 배터리의 경우, 데이터 수가 159개(평균 2,314 개)뿐

③ Various Dataset Environment
- Figure 2 와 같이 다양한 환경에서 촬영된 이미지





# 문제에 대한 실험

## 실험 결과

### Issue 및 성능 개선을 위한 시도

① Data Cleansing : Figure 3과 같은 train image의 잘못된 labeling이나 annotation을 수정해 성능 향상을 요함  
② Data Augmentation : Class Imbalance 및 Image의 촬영 환경 보완을 위한 다양한 Augmentation 기법 시도  
→ Randomfog, Blur, RandomBrightness, Cutmix, Mixup, Mosaic, Resize, normalization, MultiScale, RandomFlip  
③ Model Selection : Inductive bias를 최소화하기 위해 다양한 backbone model을 사용하여 학습  
④ Generalization : 여러 Augmentation과 Noise를 넣어 시도,  TTA시도  
⑤ Pseudo Labeling: 학습한 모델로 test 데이터를 inference한 후, 그 결과로 추가 학습  
⑥ Ensemble : 1-stage model 과 2-stage model을 Ensemble 함으로서 Robust한 모델 개선 시도  
⑦ Binary Classification : 각각의 single class를 binary classification 로 학습  


## 실험 히스토리
<p align="center"><b>Wandb를 활용한 실험 관리</b></p>

![](https://i.imgur.com/D44tEOj.png)

<p align="center">Yolo-V5 augmentation test</p>

![](https://i.imgur.com/o9KD9yU.png)

<p align="center">UniverseNet101 augmentation test</p>

![](https://i.imgur.com/h5pEg83.png)


## 검증 전략
- class별로 AP를 확인하면서 어떤 유형을 틀리고 있는지 파악
- Stratified validation과 Confusion Matrix를 통해 분류 모델 평가


<p align="center"><img src="https://i.imgur.com/xK3RTJ3.png"></p>


<br>
<br>


<div>
<center><img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fe34061b-1c71-4d04-9e68-94c8a462e7d9/confusion_matrix.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211016%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211016T113908Z&X-Amz-Expires=86400&X-Amz-Signature=fe612e73417194c63f47e5655c597bb2c726182b55ac2cc9f89f9ec9a2ac2a71&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22confusion_matrix.png%22"></center>


</div>





# Modeling 및 Ensemble

<table class="c76"><tbody><tr class="c54"><td class="c63" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>Model</b></span></p></td><td class="c82" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>개선 시도</b></span></p></td><td class="c64" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>mAP</b></span></p></td></tr><tr class="c5"><td class="c17" colspan="1" rowspan="8"><p class="c2"><span class="c3">UniverseNet101</span></p></td><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Mixup</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.573</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Mosaic / Adam / batch_size 16 / LR 0.0001 / Cutmix Dataset</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.419</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Mosaic / Adam / batch_size 16 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.451</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">HueStauration / Adam / batch_size 16 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.619</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">&nbsp;RandomFog / batch_size 16 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.623</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">RandomBrightness / batch_size 16 / LR 0.002</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>0.624</b></span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Blur, RandomFog, RandomBrightness, Mixup / Adam / batch_size 16 / LR 0.0001 /Data Cleaning</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.613</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Data cleaning / RandomBrightness / batch_size 16 / LR 0.002</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.590</span></p></td></tr><tr class="c5"><td class="c17" colspan="1" rowspan="5"><p class="c2"><span class="c3">Swin-T,S</span></p></td><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Swin-T / Adam / batch_size 4 / LR 0.0001 </span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.523</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Swin-T / MultiScale / Adam / batch_size 4 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.550</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Swin-S / MultiScale / AdamW / batch_size 4 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.561</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Swin-S / MultiScale, Mixup / AdamW / batch_size 4 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.571</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">Swin-S / Randomfog, RandomBrightnessContrast, ShiftScaleRotate, MultiScale, Mixup / AdamW / batch_size 4 / LR 0.0001</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>0.586</b></span></p></td></tr><tr class="c5"><td class="c17" colspan="1" rowspan="5"><p class="c1"><span class="c3"></span></p><p class="c2"><span class="c3">Yolo-V5</span></p></td><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">baseline (Yolo v5x6 default) / 300 epoch</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>0.586</b></span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">label smoothing (T=0.05) / 300 epoch</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.583</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">single class / 300 epoch </span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.008</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">random fog / 50 epoch</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.573</span></p></td></tr><tr class="c5"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">pseudo labeling (conf_threshold=0.6) / train data(10epoch x 3), pseudo data(10epoch x 2) 번갈아가며 학습 / pseudo data 학습시 obj loss 제외 / 50 epoch</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.559</span></p></td></tr><tr class="c35"><td class="c17" colspan="1" rowspan="3"><p class="c2"><span class="c13">PVTv2-B3</span></p></td><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c13">baseline epoch 30 / batch size 4 / scheduler step / LR 0.002 / </span><span class="c97">fp16</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.541</span></p></td></tr><tr class="c35"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">batch size 3 / LR 0.0001 / &nbsp;Adam / score_thr = 0</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c0"><b>0.573</b></span></p></td></tr><tr class="c35"><td class="c8" colspan="1" rowspan="1"><p class="c2"><span class="c3">batch size 2 / LR 0.0001/ mixup, RandomFog, blur, randbright / Adam / Data cleaning</span></p></td><td class="c23" colspan="1" rowspan="1"><p class="c2"><span class="c3">0.526</span></p></td></tr></tbody></table>

더 자세한 기록은 [더보기...](https://docs.google.com/spreadsheets/d/1xw11I8pUZY8CGaE0jXeE4KGokvK-zbpxHdKKYsyt_SI/edit#gid=265349216)

## 최종 Ensemble된 모형
<p align="center"><img src="https://i.imgur.com/2BvTb4c.png"></p>
각 model별로 가장 LB score가 좋았던 버전 5개를 Ensemble하여 최종 모델을 생성  

# 회고

## 잘했던 점
- mmdetection 라이브러리를 활용하여 object detection의 전반적인 task를 수행해 볼 수 있었습니다.
- mAP, NMS, WBF등 object detection과 관련된 용어들에 대해 공부 할 수 있었습니다.
- object detection에 관련된 다른 대회들(Kaggle, Dacon)을 찾아보고 대회에서 활용된 기법들을 이번 대회에 적용시켜 봄으로써 성능 변화나 실제로 적용시켰을 때의 어려움 등을 경험해보고 이와 관련된 내용으로 팀원들과 소통하여 성장을 도모했습니다.
- mmdetection 외에 yolov5, Universenet과 같은 다른 라이브러리를 사용해 보는 등  다양한 시도를 하였습니다.

![](https://i.imgur.com/LVSLYC5.png)


## 아쉬웠던 점

- mmdetection과 관련된 다양한 라이브러리를 처음 활용함에 어려움이 있어서 내가 원하던 모델을 구현하는데 시간이 오래 걸렸습니다. 새롭게 배워간다는 느낌보다 mmdetection 라이브러리를 통해 기계적으로 파라미터 튜닝을 하여 성능을 향상시킨다는 느낌이 강해서 아쉬웠습니다.
- object detection을 처음 입문함에 있어서 다양한 테스트를 해봐야 하는데 결과를 보기까지의 시간이 너무 오래 걸렸고, 이로 인해 다양한 실험을 기간 내에 해결하지 못했습니다.
- test dataset과 유사한 validation dataset을 찾기 어려워 연구가 진행될수록 model을 개선시키기 위한 뚜렷한 판단의 근거가 부족했습니다.
- LB mAP를 높이기 위해서는 bounding box를 많이 치는 것이 효과가 있었고 그러다 보니 도출된 결과를 확인했을때 이걸 detection했다고 할 수 있는지 의문이 들었고 mAP가 평가 지표로 적합한가 의문이 들었습니다.


<table align="center">
    <tr>
        <td><img width="340" src="https://i.imgur.com/IjbLt9r.jpg" /><br/>
            <p align='center'> mAP 0.672</p>
        </td>
        <td><img width="300" src="https://i.imgur.com/OuI6iNY.png" /><br/>
            <p align='center'> mAP 0.482</p>
        </td>
    </tr>
</table>

## 팀원들의 한마디

- 한건우 : 외부 라이브러리를 사용할 때는 항상 검증하고 사용해야한다는 점을 다시 한번 느끼게 되었습니다.
- 박준혁 : 첫 object detection 대회를 경험함으로서 많은 어려움이 있었으나 팀원들과의 소통으로 잘 해결해나갔고, 앞으로도 이런 object detection task에 대한 감을 얻을 수 있어서 좋았습니다. 팀원 모두 대회를 열심히 진행해주고 같이 고민해주어서 감동이였어요~!
- 홍요한 : 라이브러리에 익숙해지는데 시간이 오래 걸려 더 많은 실험을 못했는데 다음 대회에는 더 많은 시도를 해보고 싶습니다!
- 황원상 : 이제 감을 잡아가는 것 같습니다. 다음 대회 때는 날아가봅시다.
- 박범수 : 두번째 competition이지만 아직도 부족한게 많다고 느끼는 대회였습니다. 앞으로는 조금 더 체계적으로 대회를 진행해야겠다는 걸 깨달았습니다.
- 서희수 : 협업을 중요하다고 생각해 왔는데 다양한 라이브러리로 실험을 진행해야 하는 상황에서는 우선시 하지 않을 필요도 있다고 깨닫게 되었고, 전반적인 detection task를 알아갈 수 있었다는 점에서 의미 있지만 라이브러리 튜닝만 하다 끝났다고 생각이 들기도 해서 아쉬운 마음이 남습니다. 모두 수고 많으셨어요!
- 조혜원 : Object detection 모델들을 직접 학습해볼 수 있어서 좋았으나, 모델 구조에 대해 깊이있게 파악하지 못한 등의 아쉬움도 많이 남는 대회였습니다. 팀원분들 모두 수고많으셨어요!

# Reference

<p><span style="background-color:#EEEEEE;">네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0<br/>
https://stages.ai/competitions/76/overview/description
</span></p>

