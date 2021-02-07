---
title: Kaggle API installation
published: true
---

출처: https://github.com/Kaggle/kaggle-api

설명하는 내용은 윈도우10에 기반함.

준비물: pip, kaggle.json

쉘에서 kaggle API를 이용하기 위해서는 위 두 가지 준비물이 필요함

쉘에서 설치

```bash
 C:>pip3 install kaggle
```

버전 확인
```bash
 C:>kaggle --version
```

업그레이드
```bash
 C:>pip3 install kaggle --upgrade
```

kaggle.json을 얻기 위해서는 kaggle.com의 계정(account)에서
API항목의 'Create New API Token'을 눌러 kaggle.json을 다운받는다

kaggle.json파일을 .kaggle폴더로 이동한다.
예)
```bash
 C:>Users\username\.kaggle\kaggle.json
```

이후 실행 가능함.

실행 명령 목록들

```bash
kaggle competitions {list, files, download, submit, submissions, leaderboard}
kaggle datasets {list, files, download, create, version, init}
kaggle kernels {list, init, push, pull, output, status}
kaggle config {view, set, unset}
```

예) 2021-1-20
```bash
C:>kaggle competitions list
ref                                            deadline             category            reward  teamCount  userHasEntered
---------------------------------------------  -------------------  ---------------  ---------  ---------  --------------
contradictory-my-dear-watson                   2030-07-01 23:59:00  Getting Started     Prizes        107           False
gan-getting-started                            2030-07-01 23:59:00  Getting Started     Prizes        215           False
tpu-getting-started                            2030-06-03 23:59:00  Getting Started  Knowledge        422           False
digit-recognizer                               2030-01-01 00:00:00  Getting Started  Knowledge       2893           False
titanic                                        2030-01-01 00:00:00  Getting Started  Knowledge      22321            True
house-prices-advanced-regression-techniques    2030-01-01 00:00:00  Getting Started  Knowledge       5955           False
connectx                                       2030-01-01 00:00:00  Getting Started  Knowledge        523           False
nlp-getting-started                            2030-01-01 00:00:00  Getting Started  Knowledge       1557           False
competitive-data-science-predict-future-sales  2022-12-31 23:59:00  Playground           Kudos      10174           False
vinbigdata-chest-xray-abnormalities-detection  2021-03-30 23:59:00  Featured           $50,000        339            True
hubmap-kidney-segmentation                     2021-03-25 23:59:00  Research           $60,000        889           False
ranzcr-clip-catheter-line-classification       2021-03-15 23:59:00  Featured           $50,000        558           False
jane-street-market-prediction                  2021-02-22 23:59:00  Featured          $100,000       2796           False
cassava-leaf-disease-classification            2021-02-18 23:59:00  Research           $18,000       2916            True
rfcx-species-audio-detection                   2021-02-17 23:59:00  Research           $15,000        878           False
acea-water-prediction                          2021-02-17 23:59:00  Analytics          $25,000          0           False
rock-paper-scissors                            2021-02-01 23:59:00  Playground          Prizes       1517           False
santa-2020                                     2021-02-01 23:59:00  Featured            Prizes        760           False
tabular-playground-series-jan-2021             2021-01-31 23:59:00  Playground            Swag       1209           False
nfl-big-data-bowl-2021                         2021-01-07 23:59:00  Analytics         $100,000          0           False
```

# Colab에서 사용하기

```bash
!pip install kaggle

from google.colab import files
files.upload()
```
아래 파일선택을 클릭하고 kaggle.json파일 

이후 폴더 이동
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
