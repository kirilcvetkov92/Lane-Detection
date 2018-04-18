# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

### My pipeline consisted of the following steps:
1. Image Transformations for extracting useful structural information from the image and dramatically reduce the amount of data to be processed: 
	1.1 Transforming the image to gray-scale
	1.2 Applying Gaussian blur to smooth the image in order to remove the noise
        1.3 Applying Canny transform for edge detection
2. Extracting Mask from region of interest, the region that will contain the lane lines
3. Applying Hough transformation on the extracted mask in order to get all the candidate lines contained in the region of interest
        3.1 Line noise removal 
	    After applying hough transform, we obtain the lines formed from the edges.
	    A line `y=mx+b` is represented as (m,b), where `m` is the slope and `b` is the intercept.
            As we can see from the `Pic.1`, if we plot all the candidate lines as a points in 2d Axis,
	    there may be a noise points, simplified we can consider them as a points that are not representing the lane lines, 
	    which means their slope and intercept differs a far away from the mean.
            Below is the proposed formula for calculating the outliers.
         3.2 Adding the new lines inside the frame buffer
             Once we cleaned the noise lines from the candidate lines, we should see how much they fit
	     with all the previous lines which were inserted in the past, inside the buffer.
             If the candidate line differs a lot from the lines in the buffer, we should
             exclude the candidate, otherwise we should include them in the buffer.
         3.3 We get the last 20 lines inserted from the buffer, and we can construct the average line
             3.3.1 Mean `slope` calculation
	         3.2 Mean `intersect` calculation
     
         3.4 We plot the line on our current mask
4. We merge the mask together with our picture

You can see the image below, describing more detailed activity flow diagram of my current pipeline.


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

Potential shortcomings :
This pipeline is not robust to the following conditions :

1.If there is huge amount of noise in the region of interest (edges which are not produced from the lanes), the algorithm cannot decide which one represent the lines
2.Drastic changes in the lanes direction, if we are driving around twisty roads, this system won't be capable to detect the lines, since we are using averages from the past frames,
  and also the drastic changes will be considered as an outliers(noise)

### 3. Suggest possible improvements to your pipeline

A possible improvement would using of non-linear models that can learn how one lane is represented in the road, considering not only the edges, but many features as well.
I think that Neural networks, especially Convolutional Neural Networks can be trained to get all the regions(anchors) representing the lanes, so we can 
draw spline between the anchors, with using spline interpolations or similar numeric methods.
I think that this model will solve the shortcomings mentioned before, potential improvement is that we can train the model with data generated from different conditions with different type of noise,
so our system can be more robust and if properly trained, it can adapt to all drastic changes that can happen.
