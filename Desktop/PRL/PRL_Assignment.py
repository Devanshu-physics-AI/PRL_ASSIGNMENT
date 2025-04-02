import numpy as np
import cv2
from astropy.io import fits as ft

#load the image
img=ft.open("/users/devanshusharma/downloads/SIP_USO_PRL/UDAI.FDGB.03062019.080021.864.fits") #string contain path of image in my computer
img=img[0].data #taken first values because that contain the image information


#Since this if .fit image, normalising it to open in computer and extracting the visible features
#The formula of min max normalisation is ((pixcel-minimum_pixcel)/(maximum_pixcel)-(minimum_pixcel))*255
img=(img-np.min(img))/((np.max(img)-np.min(img)))*255
# print(img)

#Image is in floating point so converting it into the range of [0,255]
img=np.uint8(img)
img_shape=img.shape  #[2048,2048]
# print(img_shape)
# print(img)

##"OUR PROVIDED IMAGES DOESN'T CONTAIN THE NAN VALUES"##
#Canny Edge detector for edge detection with the therehold values [45,85]
boundary_img=cv2.Canny(img,45,85)
cv2.imwrite("/users/devanshusharma/desktop/PRL/image2/boundary_image.jpg",boundary_img)

#Since our image contain only the one Circle image of sun so using the HOUGH TRANSFORMATION on it
#using gradient hough because where the intensity of pixcel is changin maximum can be considered as the edge or boundary
det_circles= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,1000,param1=120,param2=50,minRadius=300,maxRadius=2000)


#Again circle is in the float format so convertin them into the interger we get
det_circles=np.uint16(np.around(det_circles))
# print(type(det_circles))  #checking type of the output- numpy array
print(str.format("The [x_coor,y_coor,radius is] {}",det_circles[0][0]))
det_circles=det_circles[0][0]



#Creatign the binary image with the sun detected
binary_img=cv2.circle(boundary_img,det_circles[0:2],det_circles[2],(255),-1)
#highligting the centre
binary_img=cv2.circle(binary_img,det_circles[0:2],7,(0),-1)
cv2.imwrite("/users/devanshusharma/desktop/PRL/image2/binary_image.jpg",binary_img)

#highlighting the sun in real image
img=cv2.circle(img,det_circles[0:2],det_circles[2],(255,255,255))
img=cv2.circle(img,det_circles[0:2],7,(0),-1)
cv2.imwrite("/users/devanshusharma/desktop/PRL/image2/detected_image.jpg",img)





cv2.imshow("boundary image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()