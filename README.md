# PD_Predictor
Classification model to classify patients with Parkinson's disease from normal people

<h3>Motivation</h3>
  <ul>
  <li>고가의 장비, 고급 인력, 복잡한 검사과정을 거치지 않고 피험자가 파킨슨 병을 앓고 있는지 확인하기 위해 거리 및 압력 센서를 이용해 보고자 하였다.</li>
  <li>최종 목표는 피험자의 앉았다 일어나기(sit-to-stand) 영상으로부터 유용한 feature를 추출해 정상군과 PD군을 분류할 수 있는 모델을 개발하는 것이다.</li>
  </ul>

<h3>Data</h3>
![CG](CG.png)
![PD](PD.png)

<h3>Progress</h3>
  <ul>
  <li>현재로써는 sit-to-stand동작을 통해 취득한 데이터로부터 시간적 특성과 압력 특성을 추출해 모델의 input으로 이용하였다.</li>
  <li>정상군의 데이터가 부족한 상황이었지만 총 5회 진행하는 데이터를 1회에 해당하는 segment를 3개씩 묶어서 데이터를 어느정도 보충 할 수 있었으며, SMOTE라는 오버샘플링 기법을
  이용해 데이터 부족 문제를 해결하였다.</li>
  </ul>
  
<h3>Model & Accuracy</h3>
  <ol>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  </ol>
  
