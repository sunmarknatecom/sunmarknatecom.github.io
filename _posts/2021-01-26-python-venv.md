---
layout: post
title: python venv and pip
categories: [python, venv, pip]
tags: [python]
published: true	
---

1. 파이썬의 가상환경

python3에서는 기본 내장 모듈: venv

ref) https://docs.python.org/ko/3.7/library/venv.html

path는 가상환경이 설치되어 있는 폴더를 말한다.

-------------------------------------------------------------------------------
1단계: 가상환경 생성
```bash
 C:\path\my_project>python -m venv 가상환경이름
```

Tip: 전역패키지 기본 설치
```bash
 C:\path\my_project>python -m venv 가상환경이름 --system-site-packages
```
-------------------------------------------------------------------------------
2단계: 가상환경 활성화
```bash
 C:\path\my_project>가상환경이름\Scripts\activate.bat
```

Tip: 파이썬 인터프리터 위치 확인
```bash
 C:\path\my_project>where python
```
첫 번째 결과는 가상환경 내 파이썬 위치
두 번째 결과는 기본 파이썬 위치

Tip: Scripts 폴더로 이동하지 않고 실행하는 방법
```bash
 C:\path>가상환경이름\Scripts\activate.bat
```
-------------------------------------------------------------------------------
3단계: 비활성화

```bash
 C:\path>deactivate
```
-------------------------------------------------------------------------------

2. pip 사용법
```bash
    C:>pip list
```
```bash
    C:>pip install
```
```bash
    C:>pip freeze > req.txt
```
```bash
    C:>pip install -r req.txt
```
3. 
