---
layout: post
title: Structured or Unstructured data
---

 1. 정형데이터 (Structured data)

 정형 데이터는 데이터베이스의 정해진 규칙(Rule)에 맞게 데이터를 들어간 데이터 중에 수치 만으로 의미 파악이 쉬운 데이터들을 보통 말한다.

 예를 들어, Gender라는 컬럼이 있는데 여기서 male, female이라는 값이 들어간다면 누가 봐도 그 값은 남자, 여자라는 것을 쉽게 인식할 수 있고, Age에 25 혹은 40과 같은 숫자가 들어가도 사람들은 쉽게 그 값을 인식할 수 있다.

 그래서 어떤 곳은 정형 데이터를 데이터베이스에 들어간 데이터라고 말하는 오류를 범하게 되는데 데이터베이스에는 모든 데이터가 들어갈 수 있는 것(음성이든, 비디오도 객체 방식으로 넣을 수 있다)이기에 이런 정의는 틀렸다고 볼 수 있다.

 정형 데이터는 그 값이 의미를 파악하기 쉽고, 규칙적인 값으로 데이터가 들어갈 경우 정형 데이터라고 인식하면 될 것이다.
  

 2. 비정형데이터(Unstructured data)

 비정형 데이터는 정형 데이터와 반대되는 단어이다. 즉, 정해진 규칙이 없어서 값의 의미를 쉽게 파악하기 힘든 경우이다. 흔히, 텍스트, 음성, 영상과 같은 데이터가 비정형 데이터 범위에 속해있다. 그래서 빅데이터의 탄생에 비정형 데이터의 역할이 크게 한 몫한 이유가, 그동안 의미를 분석하기 힘들었던 대용량에 속한 비정형 데이터를 분석함으로써 새로운 인사이트를 얻게 되기 때문이었다.

 그렇다고 빅데이터가 비정형 데이터만 분석한다는 것은 당연히 아니다. 3V에 Velocity(속도), Volume(양), Variety(다양)가 있는 것처럼 비정형 데이터는 Variety에 속하며 대용량의 정형 데이터도 얼마든지 많기 때문이다.

 3. 반정형데이터(Semi-structured data)
 
 반정형 데이터의 반은 Semi를 말한다. 즉 완전한 정형이 아니라 약한 정형 데이터라는 것이다. 대표적으로 HTML이나 XML과 같은 포맷을 반정형 데이터의 범위에 넣을 수 있을 것이다.

 일반적인 데이터 베이스는 아니지만 스키마를 가지고 있는 형태이다. 그런데 사실 반정형이라는 말이 참 까다로운 것이 데이터베이스의 데이터를 Dump하여 JSON이나 XML형태의 포맷으로 변경하면 이 순간 반정형 데이터가 된다는 것인데 쉽게 납득이 되질 않게 된다.

 그래서 한가지를 더 이해하면 되는데, 데이터베이스의 구조와 JSON이나 XML 데이터의 구조를 한번 이해해보는 것이다. 일반적으로 데이터베이스는 데이터를 저장하는 장소와 스키마가 분리되어 있어서 테이블을 생성하고, 데이터를 저장한다는 매커니즘으로 되어 있다.

 그러나 반정형은 한 텍스트 파일에 Column과 Value를 모두 출력하게 된다.

출처: [https://needjarvis.tistory.com/502](https://needjarvis.tistory.com/502) [자비스가 필요해]