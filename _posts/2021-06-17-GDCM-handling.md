---
layout: post
title: GDCM dicom instance handling
categories: [C++, GDCM]
tags: [C++, GDCM]
published: true	
---	


gdcm 인스턴스 핸들링 참고

https://stackoverflow.com/questions/30066695/gdcm-id-like-to-do-a-simple-sharpening-filter-to-the-image-but-have-no-idea-h

<code>

#include "gdcmPhotometricInterpretation.h"
#include <iostream>
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmBitmapToBitmapFilter.h"
#include "gdcmImageToImageFilter.h"

using namespace gdcm;
using namespace std;

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << argv[0] << " input.dcm output.dcm" << std::endl;

        return 1;
    }

    const char *filename = argv[1];       //name of read-in file
    const char *outfilename = argv[2];    //name of write-out file

    // Instanciate the image reader
    gdcm::ImageReader reader;
    reader.SetFileName(filename);
    if (!reader.Read()) {
        cerr << "Could not read: " << filename << endl;

        return 1;
    }

    const Image &image = reader.GetImage();

    //Get some properties from the image
    //Dimension of the image
    unsigned int n_dim = image.GetNumberOfDimensions();
    const unsigned int *dims = image.GetDimensions();
    //Origin
    const double *origin = image.GetOrigin();
    const PhotometricInterpretation &pl = image.GetPhotometricInterpretation();

    for (unsigned int i = 0; i < n_dim; ++i) {
        std::cout << "Dim(" << i << "): " << dims[i] << std::endl;
    }

    for (unsigned int i = 0; i < n_dim; ++i) {
        cout << "Origin(" << i << "): " << origin[i] << endl;
    }
    std::cout << "PhotometricInterpretation: " << pl << endl;

// The output of gdcm::Reader is a gdcm::File
    gdcm::File &file = reader.GetFile();

// the dataset is the the set of element we are interested in:
    gdcm::DataSet &ds = file.GetDataSet();


    const unsigned int *dimension = image.GetDimensions();
    unsigned int dimX = dimension[0];
    unsigned int dimY = dimension[1];
    PixelFormat pf = image.GetPixelFormat();
    unsigned long len = image.GetBufferLength();
    char *buffer = new char[len];

    image.GetBuffer(buffer);

    /*char * p = buffer;
    double temp;
    int ybr2[3];
    for (int r = 0; r < dimX; ++r)
        for (int g = 0; g < dimY; ++g)
             {

                ybr2[0] = r;
                ybr2[1] = g;

                //*p++ = (char) ybr2[0];
                *p++ = (char) ybr2[1];

            }*/

    DataElement pixeldata = image.GetDataElement();

    pixeldata.SetByteValue(buffer, len);
    delete[] buffer;
    SmartPointer<Image> im = image;
    im->SetDataElement(pixeldata);

    gdcm::ImageWriter WriterNew;
    //WriterNew.SetImage(image);
    WriterNew.SetImage(*im);
    WriterNew.SetFileName(outfilename);

    if (!WriterNew.Write()) {
        std::cerr << "Could not write: " << outfilename << std::endl;
        return 1;
    }

    return 0;
}

</code>
