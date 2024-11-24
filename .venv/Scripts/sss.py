# Load the image
import cv2
import imageio.v3 as iio
from numpy import array
import numpy as np
from PIL import Image
k57 = 206265/3600

#chop images in size
#----------------------------------------------------------------
can0 = iio.imread('acanvas0.png')

a = array(can0)
h,w, _ = a.shape
can0 = can0[:h//2,:w]

iio.imwrite('ac0.png', can0)
#----------------------------------------------------------------
can0 = iio.imread('acanvas1.png')

a = array(can0)
h,w, _ = a.shape
can0 = can0[:h//2,:w]

iio.imwrite('ac1.png', can0)
#----------------------------------------------------------------
can0 = iio.imread('acanvas2.png')

a = array(can0)
h,w, _ = a.shape
can0 = can0[:h//2,:w]
#----------------------------------------------------------------
iio.imwrite('ac2.png', can0)

can0 = iio.imread('acanvas3.png')

a = array(can0)
h,w, _ = a.shape
can0 = can0[:h//2,:w]

iio.imwrite('ac3.png', can0)
#----------------------------------------------------------------

#desintigrate the sky
#----------------------------------------------------------------
image = cv2.imread('ac0.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.ones(image.shape[:2], dtype="uint8") * 255

cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)
mask = 255 - mask
result = cv2.bitwise_and(image, image, mask=mask)

# Make a True/False mask of pixels whose BGR values sum to more than zero
alpha = np.sum(result, axis=-1) > 0

# Convert True/False to 0/255 and change type to "uint8" to match "na"
alpha = np.uint8(alpha * 255)

# Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
res = np.dstack((result, alpha))

# Save result
cv2.imwrite('ac0.png', res)
way0 = np.array(res)
#----------------------------------------------------------------
image = cv2.imread('ac1.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.ones(image.shape[:2], dtype="uint8") * 255

cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)
mask = 255 - mask
result = cv2.bitwise_and(image, image, mask=mask)

# Make a True/False mask of pixels whose BGR values sum to more than zero
alpha = np.sum(result, axis=-1) > 0

# Convert True/False to 0/255 and change type to "uint8" to match "na"
alpha = np.uint8(alpha * 255)

# Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
res = np.dstack((result, alpha))

# Save result
cv2.imwrite('ac1.png', res)
way1 = np.array(res)
#----------------------------------------------------------------
image = cv2.imread('ac2.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.ones(image.shape[:2], dtype="uint8") * 255

cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)
mask = 255 - mask
result = cv2.bitwise_and(image, image, mask=mask)

# Make a True/False mask of pixels whose BGR values sum to more than zero
alpha = np.sum(result, axis=-1) > 0

# Convert True/False to 0/255 and change type to "uint8" to match "na"
alpha = np.uint8(alpha * 255)

# Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
res = np.dstack((result, alpha))

# Save result
cv2.imwrite('ac2.png', res)
way2 = np.array(res)
#----------------------------------------------------------------
image = cv2.imread('ac3.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.ones(image.shape[:2], dtype="uint8") * 255

cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)
mask = 255 - mask
result = cv2.bitwise_and(image, image, mask=mask)

# Make a True/False mask of pixels whose BGR values sum to more than zero
alpha = np.sum(result, axis=-1) > 0

# Convert True/False to 0/255 and change type to "uint8" to match "na"
alpha = np.uint8(alpha * 255)

# Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
res = np.dstack((result, alpha))

# Save result
cv2.imwrite('ac3.png', res)
way3 = np.array(res)
#----------------------------------------------------------------







#piece all images
#----------------------------------------------------------------
for i in range(45,134):
    for j in range(0,34):
        A0 = 90/k57
        x1 = int((np.tan(i/k57-A0)*461)//1)

        y1 = int((np.tan(j/k57)*(x1**2+461**2)**0.5)//1)
        x2 = int((np.tan(i/k57-A0+1/k57)*461)//1)
        y2 = int((np.tan(j/k57+1/k57)*(x2**2+461**2)**0.5)//1)

        x1 = x1 + 461
        x2 = x2 + 461

        y1 = 441-y1
        y2 = 441-y2

        name = 'downloads/'+str(i)+'!'+str(j)+'!'+str(i+1)+'!'+str(j+1)+'.png'
        sps = way0[y2:y1,x1:x2]

        im = Image.fromarray((sps).astype(np.uint8))
        im.save(name)
#----------------------------------------------------------------
for i in range(135,224):
    for j in range(0,34):
        A0 = 180/k57
        x1 = int((np.tan(i/k57-A0)*461)//1)

        y1 = int((np.tan(j/k57)*(x1**2+461**2)**0.5)//1)
        x2 = int((np.tan(i/k57-A0+1/k57)*461)//1)
        y2 = int((np.tan(j/k57+1/k57)*(x2**2+461**2)**0.5)//1)

        x1 = x1 + 461
        x2 = x2 + 461

        y1 = 441-y1
        y2 = 441-y2

        name = 'downloads/'+str(i)+'!'+str(j)+'!'+str(i+1)+'!'+str(j+1)+'.png'
        sps = way1[y2:y1,x1:x2]

        im = Image.fromarray((sps).astype(np.uint8))
        im.save(name)
#----------------------------------------------------------------
for i in range(225,314):
    for j in range(0,34):
        A0 = 270/k57
        x1 = int((np.tan(i/k57-A0)*461)//1)

        y1 = int((np.tan(j/k57)*(x1**2+461**2)**0.5)//1)
        x2 = int((np.tan(i/k57-A0+1/k57)*461)//1)
        y2 = int((np.tan(j/k57+1/k57)*(x2**2+461**2)**0.5)//1)

        x1 = x1 + 461
        x2 = x2 + 461

        y1 = 441-y1
        y2 = 441-y2

        name = 'downloads/'+str(i)+'!'+str(j)+'!'+str(i+1)+'!'+str(j+1)+'.png'
        sps = way2[y2:y1,x1:x2]

        im = Image.fromarray((sps).astype(np.uint8))
        im.save(name)
#----------------------------------------------------------------
for i in range(315,360):
    for j in range(0,34):
        A0 = 360/k57
        x1 = int((np.tan(i/k57-A0)*461)//1)

        y1 = int((np.tan(j/k57)*(x1**2+461**2)**0.5)//1)
        x2 = int((np.tan(i/k57-A0+1/k57)*461)//1)
        y2 = int((np.tan(j/k57+1/k57)*(x2**2+461**2)**0.5)//1)

        x1 = x1 + 461
        x2 = x2 + 461

        y1 = 441-y1
        y2 = 441-y2

        name = 'downloads/'+str(i)+'!'+str(j)+'!'+str(i+1)+'!'+str(j+1)+'.png'
        sps = way3[y2:y1,x1:x2]

        im = Image.fromarray((sps).astype(np.uint8))
        im.save(name)
for i in range(1,44):
    for j in range(0,34):
        A0 = 0.000000001/k57
        x1 = int((np.tan(i/k57-A0)*461)//1)

        y1 = int((np.tan(j/k57)*(x1**2+461**2)**0.5)//1)
        x2 = int((np.tan(i/k57-A0+1/k57)*461)//1)
        y2 = int((np.tan(j/k57+1/k57)*(x2**2+461**2)**0.5)//1)

        x1 = x1 + 461
        x2 = x2 + 461

        y1 = 441-y1
        y2 = 441-y2

        name = 'downloads/'+str(i)+'!'+str(j)+'!'+str(i+1)+'!'+str(j+1)+'.png'
        sps = way3[y2:y1,x1:x2]

        im = Image.fromarray((sps).astype(np.uint8))
        im.save(name)