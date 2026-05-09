# Implementation of Sobel Filter using CUDA

**Akshita Akasapu¹, V Sailaja², G Prasad³**

¹&² Department of EEE, Amrita School of Engineering, Bengaluru, Amrita Vishwa Vidyapeetham, India.  
³ Special Projects and Photo processing division, National Remote Sensing Center, Indian Space Research Organization, Hyderabad, India.

**Emails:** akshitaakasapu@gmail.com, v_sailaja@blr.amrita.edu, prasad_g@nrsc.gov.in

---

## Abstract

The Sobel filter is an edge detection filter that results in highlighting and focusing the edges. It is used for extracting the image gradient and contours in image processing applications. The data sets we are using in the paper are **PGM** and **TIFF** images with different rows and column pixel dimensions. The Sobel filter operations used are implemented in **C** and **CUDA** programming language and the time taken for the operations in the serial and parallel platforms respectively are analyzed.

**Keywords:** Image Processing, Edge Detection, Sobel operator, C programming language, CUDA programming language, Parallel Performance, Serial Performance, PGM file, TIFF file.

---

## 1. Introduction

Image processing is one of the most powerful tools used in fields with requirements of transmission of data especially in telecommunication, security systems, and remote sensing. ...

Satellite sensors collect data in different spectral bands and hence this data is represented in many dimensions. Due to the multiple dimensions, redundancy of data arises. To remove this and to emphasize and isolate the required data from the image, **edge detection** is applied.

Edge Detection is the general term for a technique to reduce the dimensionality of the image and isolate the required data from the huge data by emphasizing the edges.

---

## 2. Related Work

Edge detection is a foundation tool in image processing mainly focusing on feature detection and extraction. ...

The graphics processing unit (GPU) provides a constructive solution for the intensive computing of the image during image processing and provides high performance in real time applications.

---

## 3. Edge Detection Technique

The **Sobel operator** is a derivative mask and is used for edge detection. Sobel operator is used to detect the horizontal and vertical edges in an image.

### 3.1 Horizontal and Vertical Gradient Components

**Horizontal Sobel Operator (Gx):**