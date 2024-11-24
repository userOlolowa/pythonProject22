from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import sys
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import cv2

# Load the image
import imageio.v3 as iio
from numpy import array
from PIL import Image
k57 = 206265/3600
computopowar = 1 #CHANGE IT IF YOU WANT

#gogle

df = pd.read_csv('locations.csv')
fi = df['fi']
l = df['l']

z = np.random.randint(0, high=10000, size=None, dtype=int)

name = 'https://www.google.ru/maps/place/'+str(fi[z])+'+'+str(l[z])+'/'
driver = webdriver.Edge()

driver.get(name)
time.sleep(2*computopowar)


chel = driver.find_element(By.ID ,"runway-expand-button")




action = webdriver.common.action_chains.ActionChains(driver) # зажимаем человечка
action.move_to_element(chel) \
    .click_and_hold(chel) \
    .move_by_offset(0, 30) \
    .perform()

sc = driver.find_element(By.ID ,"scene")
action.move_to_element(sc) \
    .move_by_offset(0, 441) \
    .perform()

action.move_to_element(sc) \
    .move_by_offset(0, -441) \
    .perform()
time.sleep(0.052)
# time.sleep(0.1)
action.move_to_element(sc) \
    .release(sc) \
    .perform()



time.sleep(2.5*computopowar)
zooom = driver.find_element(By.ID ,"widget-zoom-out")
ActionChains(driver) \
    .click(zooom) \
    .perform()
time.sleep(0.5*computopowar)

#удаляем всякое

element = driver.find_element(By.ID, 'omnibox-container')
driver.execute_script("""
var element = arguments[0];
element.parentNode.removeChild(element);
""", element)

element = driver.find_element(By.ID, 'titlecard')
driver.execute_script("""
var element = arguments[0];
element.parentNode.removeChild(element);
""", element)



element = driver.find_element(By.ID, 'image-header')
driver.execute_script("""
var element = arguments[0];
element.parentNode.removeChild(element);
""", element)
#------------------------------------------------------------------------------------------------------------

driver.find_element(By.TAG_NAME,'canvas').screenshot('acanvas0.png')
main = driver.find_element(By.ID ,"scene")

for _ in range(3):
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element(main) \
        .click_and_hold(main) \
        .move_by_offset(258, 0) \
        .release(main) \
        .perform()
# ------------------------------------------------------------------------------------------------------------
time.sleep(0.1)
#------------------------------------------------------------------------------------------------------------

driver.find_element(By.TAG_NAME,'canvas').screenshot('acanvas1.png')
main = driver.find_element(By.ID ,"scene")

for _ in range(3):
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element(main) \
        .click_and_hold(main) \
        .move_by_offset(258, 0) \
        .release(main) \
        .perform()
# ------------------------------------------------------------------------------------------------------------
time.sleep(0.1)
#------------------------------------------------------------------------------------------------------------

driver.find_element(By.TAG_NAME,'canvas').screenshot('acanvas2.png')
main = driver.find_element(By.ID ,"scene")

for _ in range(3):
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element(main) \
        .click_and_hold(main) \
        .move_by_offset(258, 0) \
        .release(main) \
        .perform()
# ------------------------------------------------------------------------------------------------------------
time.sleep(0.1)
#------------------------------------------------------------------------------------------------------------
driver.find_element(By.TAG_NAME,'canvas').screenshot('acanvas3.png')

driver.quit()

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