---
layout: post
title: dcmtk -> cmake
categories: [C++]
tags: [C++]
published: true	
---

1. dll
C:\Users\rosie\Desktop\source\dcmtk-3.6.6_dst\bin
2. lib
C:\Users\rosie\Desktop\source\dcmtk-3.6.6_dst\lib
3. .h
C:\Users\rosie\Desktop\source\dcmtk-3.6.6_dst\include

참고:
Cmake에서 cmake prefix를 설치폴더(예: C:\Users\rosie\Desktop\source\dcmtk-3.6.6_dst)를 지정하고, shared lib를 꼭 설치

header 파일은 VS에서 빌드하고, install을 빌드해야 생성이 됨.

```C++
#include <iostream>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <errors.h>

using namespace std;

int main()
{
    DicomImage* image = new DicomImage("img.dcm");
    if (image != NULL)
    {
        if (image->getStatus() == EIS_Normal)
        {
            if (image->isMonochrome())
            {
                image->setMinMaxWindow();
                Uint8* pixelData = (Uint8*)(image->getOutputData(8 /* bits */));
                if (pixelData != NULL)
                {
                    /* do something useful with the pixel data */
                }
            }
        }
        else
            cerr << "Error: cannot load DICOM image (" << DicomImage::getString(image->getStatus()) << ")" << endl;
    }
    delete image;

	return 0;
}
```
