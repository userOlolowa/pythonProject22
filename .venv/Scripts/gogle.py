# import google_streetview
# # key= 'AIzaSyAdwOeHBStv93x0UxIievZiNALeDyBf4Tk'
# import google_streetview.api
#
# # Define parameters for street view api
# params = [{
# 	'size': '600x300', # max 640x640 pixels
# 	'location': '46.414382,10.013988',
# 	'heading': '151.78',
# 	'pitch': '-0.76',
# 	'key': 'ad8c60c5bfmsh4b2ad3d716c14d8p127154jsnf5cdcff4b12b'
# }]
#
#
# # Create a results object
# results = google_streetview.api.results(params)
#
# # Download images to directory 'downloads'
# results.download_links('downloads')

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

df = pd.read_csv('locations.csv')
fi = df['fi']
l = df['l']

z = np.random.randint(0, high=10000, size=None, dtype=int)

name = 'https://www.google.ru/maps/place/'+str(fi[z])+'+'+str(l[z])+'/'
driver = webdriver.Edge()

driver.get(name)
time.sleep(2)


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



time.sleep(2.5)
zooom = driver.find_element(By.ID ,"widget-zoom-out")
ActionChains(driver) \
    .click(zooom) \
    .perform()
time.sleep(0.5)

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

#Обработка изображений (eww)



# Load the image
image = cv2.imread('acanvas0.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to binarize the image
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask with white pixels
mask = np.ones(image.shape[:2], dtype="uint8") * 255

# Draw the contours on the mask
cv2.drawContours(mask, contours, -1, 0, thickness=cv2.FILLED)

# Perform bitwise AND on the original image and the mask
result = cv2.bitwise_and(image, image, mask=mask)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
