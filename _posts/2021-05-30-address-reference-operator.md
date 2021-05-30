---
layout: post
title: 주소 & 참조 연산자
categories: [C++]
tags: [C++]
---

C++에서 포인터를 공부하면서 주소 그리고 참조 연산자를 매일 어렵다고 한다.

주소 연산자(Address operator) : &
참조 연산자(Reference operator, pointer) : *

<- * 참조 연산자 추가하면 변수로 감 , & 주소 연산자 추가하면 주소로 감 ->

int addr = 0;
int *ptr = &addr;

R-value는 변수가 되고, L-value는 주소가 되므로, =는 사실 대입이 아님.

사실 개념상
ptr = &addr
*ptr = addr

이 맞다.
