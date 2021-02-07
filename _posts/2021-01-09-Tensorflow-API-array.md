---
title: Tensorflow - Data API - 텐서 생성
published: true
---

텐서 생성
    from_tensor_slices(): 개별 또튼 다중 넘파이를 받고, 배치를 지원
    from_tensors: 배치를 지원하지 않음
    from_generator(): 생성자 함수에서 입력을 취함

(예제)
        from_tensor_slices()

```bash
 >>> import numpy as np
 >>>
 >>> num_items = 20
 >>> num_list = np.arange(num_items)
 >>>
 >>> num_list_dataset = tf.data.Dataset.from_tensor_slices(num_list)
 >>> num_list_dataset
 <TensorSliceDataset shapes: (), types: tf.int64>
 >>> for item in num_list_dataset:
 ...     print(item)
 ...
 tf.Tensor(0, shape=(), dtype=int32)
 tf.Tensor(1, shape=(), dtype=int32)
 tf.Tensor(2, shape=(), dtype=int32)
 tf.Tensor(3, shape=(), dtype=int32)
 tf.Tensor(4, shape=(), dtype=int32)
 tf.Tensor(5, shape=(), dtype=int32)
 tf.Tensor(6, shape=(), dtype=int32)
 tf.Tensor(7, shape=(), dtype=int32)
 tf.Tensor(8, shape=(), dtype=int32)
 tf.Tensor(9, shape=(), dtype=int32)
 tf.Tensor(10, shape=(), dtype=int32)
 tf.Tensor(11, shape=(), dtype=int32)
 tf.Tensor(12, shape=(), dtype=int32)
 tf.Tensor(13, shape=(), dtype=int32)
 tf.Tensor(14, shape=(), dtype=int32)
 tf.Tensor(15, shape=(), dtype=int32)
 tf.Tensor(16, shape=(), dtype=int32)
 tf.Tensor(17, shape=(), dtype=int32)
 tf.Tensor(18, shape=(), dtype=int32)
 tf.Tensor(19, shape=(), dtype=int32)
```

변환
    batch(): 순차적으로 지정한 배치사이즈로 데이터셋을 분할
    repeat(): 데이터를 복제
    shuffle(): 데이터를 무작위로 섞음
    map(): 데이터에 함수를 적용
    filter(): 데이터를 거르고자 할 때 사용

반복
    next_batch = iterator.get_next() 사용
