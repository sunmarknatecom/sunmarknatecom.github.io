---
title: TFRecord explaination
published: true
---

출처: https://ingeec.tistory.com/89

TFRecord에 대한 설명
개요
  1. TFRecord 파일은 텐서플로의 표준 데이터 파일 포맷 (본질적으로 Protocol Buffer 파일)
  2. TFRecord 파일은 데이터를 시퀀셜하게 저장
    랜덤 억세스에 적합하지 않음
    대용량 데이터를 스트리밍 하는 데 적합
  3. 파일 하나에 모든 dataset과 label을 묶어 놓으면 파일 처리 시간이 단축됨 (권장!)

TFRecord 파일 구조

  1. TFRecord 파일은 record 들의 반복
  2. TFRecords > examples > features 형태로 구성
    TFRecord 파일은 일련의 example들(== record들)로 구성
    example은 일련의 feature들로 구성
    . https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/example.proto
    feature는 ML task 수행을 위해 필요한 데이터(ex, 입력 데이터, label 데이터)들로 구성
    . https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/feature.proto
 

TF Input Pipeline
개요
  1. TF Computation-graph에 데이터를 공급하는 방법들 중 하나
    TF 프로그램에 데이터를 공급하는 방법 3가지
      feeding
      . 매 스텝을 실행할 때 마다 python 코드가 데이터를 공급

  with tf.Session():
    input = tf.placeholder(tf.float32)
    classifier = ...
    classifier.eval(feed_dict={input: my_python_fn()}))
. 연산 과정에서 TF 그래프와 Python 사이의 context switch 필요 ==> 성능저하

input pipeline
. TF 그래프 첫머리의 input pipeline이 데이터를 가져옴 ==> 추천!

preloaded data
. TF 그래프의 variable이나 constant에 데이터를 미리 적재
==> dataset 이 작을 때만 가능

TF Input Pipeline은 모든 포맷의 파일을 사용 가능하나, TFRecord 포맷의 파일 사용을 추천

TF Input Pipeline이 하는 일

파일 목록 가져오기
파일 목록 섞기 (옵션)
파일 큐 생성하기
데이터 읽기, 데이터 디코딩 하기
이후 소개하는 "Queue based Input Pipeline"은 "Dataset API based Input Pipeline"으로 깔끔하게 대체 가능

하지만, Dataset API는 TF 1.4+에서만 사용 가능
TF 1.4는 2017-11-03 출시 (약 1달전), 아직 샘플 코드가 부족함
 

TF Input Pipeline의 전형적인 구성
(TFRecord 파일 읽기 프로세스)




Queue based Input Pipeline 동작양식

Filename Queue 생성

Filename 목록으로 Queue 생성 (파일이 1개라도 OK)
tf.train.string_input_producer() 함수 이용
. filename 목록 shuffle 옵션 제공
. filename queue 반복횟수(epoch) 설정 옵션 제공
string_input_producer() 함수는 TF 그래프에 QueueRunner 객체를 추가함
string_input_producer() 함수가 추가하는 QueueRunner는 filename queue를 구동하는 역할을 하며 Computation-graph 연산을 구동하는 QueueRunner와 별도의 스레드에서 실행되어 서로 블록되지 않는다
Reader, Decoder 정의 (또는 선택)

파일 포맷 별로 적절한 Reder를 선택하고 적절한 Decoder를 정의/선택해야 함
파일 포맷 별로 다양한 Reader 와 Decoder 제공

CSV 파일
. Reader: tf.TextLineReader
. Decoder: tf.decode_csv()
Fixed Length Record 파일
. Reader: tf.FixedLengthRecordReader
. Decoder: tf.decode_raw()
. 각 record 가 고정된 길이인 파일을 읽을 때
TF 표준 파일 (즉, TFRecord 파일) ==> TF 권장 포맷
. Reader: tf.TFRecordReader
. Decoder: tf.parse_single_example()
. 어떤 데이터이든 TFRecord 파일로 변환해서 사용할 것을 권장
Preprocessing (optional, 뭐라도 처리할 일이 있으면 실행)

Example Queue 생성/구동

pipeline 마지막 단계에서 학습/평가/추론에 batch 데이터를 공급하는 별도의 큐를 생성/운영
tf.train.shuffle_batch() 함수를 이용
. example들의 순서 난수화 가능
. batch size 설정 옵션 제공
suffle_batch() 함수는 TF Computation-graph에 QueueRunner 객체를 추가함 이 때문에 학습/추론 등을 시작할 때 tf.train.start_queue_runners()를 호출해서 input pipeline을 구동하는 스래드를 생성/실행시켜야 함.
또, 학습/추론 등이 끝날 때에는 tf.train.Coordinator 를 이용해서 관련 스래드를 모두 종료시켜야 함.
 

TFRecord 파일 저장 프로세스
TFRecord 파일 오픈

tf.python_io.TFRecordWriter
데이터를 적절한 타입으로 변환

TFRecord 파일에 저장 가능한 feature 타입의 종류

tf.train.Int64List
tf.train.BytesList
tf.train.FloatList
feature 생성

tf.train.Feature
example 생성

tf.train.Example
example을 시리얼라이즈

example.SerializeToString()
시리얼라이즈한 example을 TFRecord 파일에 기록

writer.write


출처: https://ingeec.tistory.com/89 [없으면 없는대로]
