# Book-Cover-Detection

1.	Project Description and Background

•	What is your goal? What is the problem you are trying to solve? 
o	In the application where I save the books I read, I usually have to scan the barcode on the back to find the exact version I have been reading.  I thought it would be an improved addition to make the book cover scannable by comparing it with the books in the database.

•	A few paragraphs: What is the state-of-the-art regarding your technical goal? How are other people achieving the same goal? What are the most well-known, classic methods that have been used in the past and/or are still in use?
o	It works with satisfactory %58 in my test data.
o	In other people's work, I have observed that to increase the accuracy, some extract features from an image using CNN instead of SIFT and implement text recognition algorithms to help with the problems regarding dark, one-toned images.

•	A very brief and clear explanation of your work. Explain what you’ve done in two/three sentences. Don’t get too technical, just give a brief and crisp explanation.
o	By applying different level matching systems which are RGB Color Histogram, comparing images using Structural Similarity Index Measure, and finally using SIFT features to find the closest match and display it to the user.


2.	Technical Work
•	2-4 pages: Detailed explanation of your work. Provide all necessary technical details: Algorithms, APIs, programming languages, all design and implementation details…

o	Programming Language: Python 3.10

o	Libraries: 
-	Os: To interact with the file system.
-	OpenCv: To perform operations such as adding borders, adding blurs on images.
-	NumPy: To represent images as NumPy arrays for computational efficiency.
-	SciPy: To apply affine transformation to the images.
-	Math: To calculate floor and ceil integers.
-	Matplotlib: To show my results.

o	Implementation: 

I obtained the paths of all the images in the given data-base path and added them to a list called paths.

I read the images in the paths in the list I obtained with OpenCV and added them to the list called images, and in the meantime, I obtained the maximum row and maximum column values for future use.

I set up 3-step matching, these are respectively

1.	1st match: Correlation Between RGB Histograms
I calculated the RGB color histograms of all the pictures, respectively, and added it to a list called img_hist, and then calculated it for the picture to be tested.

I compared all the pictures with the test picture and added the pictures in which correlation comparison results were higher than the correlation matrix that I fixed before to the list named matches, with the result of the comparison, the picture itself, and the path of the image. Finally, I sorted this list according to the result of the two elements, the comparison.

2.	2nd match: SSIM
During SSIM, some operations with the NumPy matrix required the matrix sizes to be equal. For this, I applied the border replicate padding by using the maximum column and maximum row values, which I have obtained before because, in my opinion, it will affect the SSIM result at least.

I calculated the SSIM score between all the photos I have from the previous match and tested image and placed the ones larger than the similarity index I determined before into the l thing named matches2 as the SSIM score, the image itself, and the photo's path, respectively. Then, I sorted the list according to the SSIM score from ascending to descending. Finally, I limited this list to only the first 50 elements for use in the next step

As for the calculation of the SSIM score, I created my own function by combining several different sources. The Structural Similarity Index metric extracts 3 key features from an image, these are luminance, contrast, and structure.

Luminance is measured by averaging over all the pixel values.
 
 ![image](https://user-images.githubusercontent.com/55497608/209398489-dcdbe0fc-d813-43f1-a075-54ce095c207c.png)

Contrast is measured by taking the standard deviation of all the pixel values.

![image](https://user-images.githubusercontent.com/55497608/209398478-8621ed86-5236-43c8-a36a-1147941bf007.png)

Structural comparison is dividing the input signal with its standard deviation.

![image](https://user-images.githubusercontent.com/55497608/209398457-da24c6b4-5c20-485b-b757-465469e1dc02.png)

And now we need the comparison functions.
Luminance Comparison Function: (L is dynamic range of pixels in my case 255 since images are 8-bits.)

![image](https://user-images.githubusercontent.com/55497608/209398428-8a1091c3-76ab-4635-8212-c37c1dace4a0.png)

![image](https://user-images.githubusercontent.com/55497608/209398438-e685f183-e95c-4e33-925a-4fb615edf088.png)

Contrast Comparison Function: 
 
![image](https://user-images.githubusercontent.com/55497608/209398411-89bdff49-d865-4c2a-9eea-1dd995af3e40.png)

![image](https://user-images.githubusercontent.com/55497608/209398417-816c8639-4e33-4cd0-a37c-da77caf591ce.png)

Structure Comparison Function:
![image](https://user-images.githubusercontent.com/55497608/209398356-dde3d02f-066d-48b4-969d-554defd5140c.png)

![image](https://user-images.githubusercontent.com/55497608/209398368-03387e57-f3ab-4fb5-bdac-1fbdced144ff.png)

SSIM score is with combining them all

![image](https://user-images.githubusercontent.com/55497608/209398391-caa13a2d-f71d-4c92-b6dd-6a6e7d8f6396.png)
			 
To simplify we assume that α = β = γ = 1 and C3 = C2/2

![image](https://user-images.githubusercontent.com/55497608/209398349-4c517e54-7f59-421b-bc41-21dd5be950ef.png)

Unfortunately, using the algorithm in this way will not work well enough, so it would be better to use the regionally rather than the globally. This method is often called as the Mean Structural Similarity Index. For this, we need to make some changes in the above formulas.

![image](https://user-images.githubusercontent.com/55497608/209398330-23a4c6b3-c43a-4583-b755-d7f1b1d4b56f.png)
 
(wi is the gaussian weighting function)

We get global SSIM value by taking mean of all local SSIM values.

![image](https://user-images.githubusercontent.com/55497608/209398313-57c0c5c3-a144-477c-bb4f-94f5aad28f5f.png)
 
3.	3rd match: SIFT
The SIFT function of OpenCV gives us key points for all our points of interest, with their x, y, and octave locations as well as their orientation. The key points are scale-invariant and rotation-invariant.

I extracted the SIFT key points and descriptors on the test image with all potential matches from the previous stage. 

I used FLANN to compare key points and quickly converge to nearest neighbor in Multidimensional space.

As a result of this step, I got the closest 3 matches and sent the image with the highest match among them as the last match.


3.	Experiments and Results
•	1-3 pages: Detailed explanation of your experimental setup and results. Provide all necessary technical details: Dataset, hyperparameters, visual or other results…

As the dataset, I chose the Science-Fiction-Fantasy-Horror file in the dataset I downloaded from kaggle.

I created a new file by choosing 50 random images from this file using the python os library to select the images to test.  

I applied affine transformation and sheer to these images to make them look like they were taken by the user. And I set the lambda value in this replacement to 0.2 because the images are small.

Looking at the results, I found it appropriate to set the correlation threshold as 0.25, if I had more data, I would have to increase this number.

Based on my observation on my SSIM function results I set the similarity index as 0.05.

I have set the number of SIFT key points to be generated as 1000.

When I try it on my own test data, I observe 56% true positive results. This percentage can be improved by tweaking the hyperparameters and consequently increasing the runtime

When I examined the mismatches, I observed that there are book covers that generally have a single tone and do not have detailed drawings etc. on them.


4.	Conclusions
•	One paragraph: Briefly explain your observations and what you have learned.
In this project, I learned to work on RGB images, calculate their histograms. Implement the SSIM algorithm and using it to compare 2 different images. And how to use SIFT features a s well as evaluating sift features with FLANN. I have observed how big an effect the change in hyperparameters has on the accuracy of the output.


5.	References
i.	https://www.kaggle.com/datasets/lukaanicin/book-covers-dataset
ii.	https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
iii.	https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb
iv.	https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

