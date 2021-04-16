---
layout: post
title: C++ language 01
categories: [C++]
tags: [C++]
---

C언어 예약어

01. auto
02. break
03. case
04. char
05. const
06. continue
07. default
08. do
09. double
10. else
11. enum
12. extern
13. float
14. for
15. goto
16. if
17. int
18. long
19. register
20. return
21. short
22. signed
23. sizeof
24. static
25. struct
26. switch
27. typedef
28. union
29. unsigned
30. void
31. volatile
32. while
33. _Bool
34. _Complex
35. _Imaginary
36. inline


C 언어의 형식문자

01. %c - int(char) : character. ASCII문자로 출력
02. %d - int : Decimal. 부호가 있는 10진수로 출력
03. %o - int : Octal. 8진수로 출력
04. %u - unsigned int : Unsigned. 부호가 없는 10진수로 출ㄺ
05. %x, %X : Hexa. 16진수로 출력
06. %e, %E - float, double : Exponent. 지수형 소수로 출력
07. %f - double(float) : Float. 10진형 소수로 출력
08. %g - double : 지수형 소수(%e)나 10진형 소수(%f)로 출력. 단, 출력되는 문자열이 짧은 형태로 출력한다.
09. %p - Pointer : 16진수 주소로 출력
10. %s - String : 인수가 가리키는 메모리의 내용을 문자열로 출력

C 언어의 이스케이프 시퀀스

01. \a : 경고음 울림
02. \b : backspace
03. \f : 인쇄시 종이 한장 넘김(Form feed)
04. \n : New line
05. \r : carriage return
06. \t : Tab
07. \v : Vertical Tab
08. \\ : backslash
09. \? : 물음표
10. \' : 작은 따옴표. 문자상수
11. \" : 큰 따옴표. 문자상수
12. \ooo : 8진수
13. \xhh : 16진수

연산자 우선순위

01. () [] . ->                         : ->
02. * & ! ++ -- (datatype) sizeof -    : <-
03. * % /                              : ->
04. + -                                : ->
05. << >>                              : ->
06. < > <= >=                          : ->
07. == !=                              : ->
08. &                                  : ->
09. ^                                  : ->
10. |                                  : ->
11. &&                                 : ->
12. ||                                 : ->
13. ?:                                 : <-
14. = += -= *= %= /= &= |= ^= <<= >>=  : <-
15. ,                                  : ->
