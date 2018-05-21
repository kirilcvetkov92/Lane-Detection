
# coding: utf-8

# 
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# # **Outlier detection:**
# 
# `reject_outliers()` for rejecting outliers from dataset 
# 
# `is_outlier()` for check whether candidate is outlier or not
# 
# 

# In[3]:


def reject_outliers(data, m=2, min_data_length=4):
    """
    The mean of a distribution will be biased by outliers.
    
    `data` Data set that represent lines as a points(hough transform) from where we want to exclude all line outliers
    
    `m` : how much we scale from standard deviation.
    
    `min_data_length` : how much entries the dataset should have before start excluding outliers
    
    The result is filtered dataset representing lines without otuliers

    """
    if len(data)>min_data_length:
        data = np.array(data)

        #filter intercept outliers
        intercepts = data[:,1]
        filter_data=data[abs(intercepts - np.mean(intercepts)) < m * np.std(intercepts)]

        #filter slope outliers
        slopes = filter_data[:,0]
        filter_data=filter_data[abs(slopes- np.mean(slopes)) < m * np.std(slopes)]
        return filter_data.tolist()
    else: 
        return data

def is_outlier(point, data, m=2, min_data_length=4):
    """
    The mean of a distribution will be biased by outliers.
     
    `point` outlier candidate point representing the line in hough transform, 
    
    `data` is the data set representing lines as a points(hough transform)
    
    `m` : how much we scale from standard deviation.
    
    `min_data_length` : how much entries the data should have before start excluding outliers
    
    The result is True if the candidate point is outlier, false otherwise

    """
    if len(data)<=min_data_length:
        return False

    data = np.array(data)
    
    slopes = data[:,0]
    intercepts = data[:,1]

    slope_diff = np.abs(point[0] - np.mean(slopes))
    intercept_diff = np.abs(point[1] - np.mean(intercepts))

    slope_std= m * np.std(slopes)
    intercept_std= m * np.std(intercepts)

    is_outlier = slope_diff >= slope_std or intercept_diff>=intercept_std

    return is_outlier


# # Mean Line
#     1. We keep the lines formed from the latest X frames into a buffer
#     2. Then we can draw the average line among all lines in the buffer

# In[4]:


def get_mean_line_from_buffer(buffer, frames, y_min, y_max):
    """
    We should keep the lines formed from the latest frames.
    Then we can draw the average line among all lines in the buffer
    
    `buffer` list containing the lines in form : (slope, intercept) from all previous frames
    
    `frames` how much frames we should consider for calculating the mean line
    
    The result is mean line in form : (slope, intercept)
    """
  
    #get the mean line from the frame buffer
    mean_line = np.mean(np.array(buffer[-frames:]), axis=0).tolist()
    mean_slope = mean_line[0]   
    mean_intercept = mean_line[1]
    
    #calculate the X coordinates of the line
    x1 = int((y_min-mean_intercept)/mean_slope)
    x2 = int((y_max-mean_intercept)/mean_slope)
    return x1, x2


# In[5]:


# buffer cache through frames
line_low_avg_cache = []
line_high_avg_cache = []

#calculate max y coordinate for drawing and constructing the line
factor_height = 0.62


# ## Helper Functions

# In[6]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global line_low_avg_cache
    global line_high_avg_cache
    
    #get min y and max y for drawing the line
    y_min = img_height = img.shape[0]
    y_max = int(img_height*factor_height)

    lines_left = []
    lines_right = []
    
    #iterate all the lines
    for line in lines:

        for x1,y1,x2,y2 in line:
            
            #calculate slope
            slope = (y2 - y1)/(x2 - x1)
            
            #exclude non-valid slopes
            if (abs(slope)<0.5 or abs(slope)>0.8):
                continue
                
            if(math.isnan(slope) or math.isinf(slope)):
                continue
             
            #calculate intercept
            intercept = y1 - x1*slope
            
            if(slope<0):
                #it's right line
                lines_left.append([slope,intercept])
                
            else :
                #it's left line
                lines_right.append([slope,intercept])

    
    #clean left lines from noise
    lines_low = reject_outliers(lines_right,m=1.7)
    
    #clean right lines from noise
    lines_high = reject_outliers(lines_left, m=1.7)
    
    
    #add left lines to the frame buffer only if they are not outliers inside
    if lines_high:
        for element in lines_high:
            if not is_outlier(element,line_high_avg_cache,m=2.6):
                line_high_avg_cache.append(element)
    
    #add right lines to the frame buffer only if they are not outliers inside
    if lines_low:
         for element in lines_low:
            if not is_outlier(element,line_low_avg_cache,m=2.6):
                line_low_avg_cache.append(element)

    if line_high_avg_cache:
        x1, x2 = get_mean_line_from_buffer(buffer=line_high_avg_cache, 
                                           frames=20,
                                           y_min = y_min,
                                           y_max = y_max)
        #line extrapolation
        cv2.line(img,(x1, y_min),(x2, y_max),color,thickness)
    
    if line_low_avg_cache:
        x1, x2 = get_mean_line_from_buffer(buffer=line_low_avg_cache, 
                                           frames=20,
                                           y_min = y_min,
                                           y_max = y_max)
        #line extrapolation
        cv2.line(img,(x1, y_min),(x2, y_max),color,thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# Below are some helper functions to help get you started. They should look familiar from the lesson!

# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[7]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# # ***Parameters***
# 
# Below are the parameters for our model pipeline:

# In[8]:


#canny edge detection params
low_threshold = 50
high_threshold = 150
kernel_size = 5

#hough transform params
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 17   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 80  #minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

#set mask scale factor
mask_scale_factor = 0.60
mask_width_factor = 0.5


# In[9]:


def detect_segments(image):
    
    #get image shape
    imshape = image.shape
    img_width = image.shape[1]
    img_height = image.shape[0]
    
    #apply greyscale transform
    grayscale_transform_image = grayscale(image)
    
    #apply gausian transform
    gausian_transform_image = gaussian_blur(grayscale_transform_image, kernel_size)
    
    #perform canny transform
    canny_transform_image = canny(gausian_transform_image, low_threshold, high_threshold
                                 )
    #get mask x, y coordinates
    mask_y = int(img_height*mask_scale_factor)
    mask_x = int(img_width*mask_width_factor)
    
    #compose mask vertices
    vertices = np.array([[(0,img_height),(mask_x, mask_y), (mask_x, mask_y), (img_width,img_height)]])
    
    #perform region of intersect
    masked_edges = region_of_interest(canny_transform_image, vertices)

    # Hough transform on edge detected image
    lines_transform_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    #merge tranformed image with the current image
    result_image = weighted_img(image, lines_transform_image)
    
    return result_image


# In[10]:


def clear_cache():
    line_low_avg_cache.clear()
    line_high_avg_cache.clear()


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[11]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[12]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = detect_segments(image)
    return result


# Let's try the one with the solid white lane on the right first ...

# In[19]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clear_cache()
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[20]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[15]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
clear_cache()
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[21]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[17]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_video|s/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
clear_cache()
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[18]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>

""".format(challenge_output))

