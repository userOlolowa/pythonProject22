import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"

import imageio.v3 as iio
import pandas as pd
import astropy as ap
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Rectangle,PopMatrix, Rotate, PushMatrix
from kivy.core.text import Label as CoreLabel
from kivy.lang import builder


from astropy import units as u
from astropy.coordinates import (SkyCoord, Distance, Galactic,
                                 EarthLocation, AltAz)
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from astropy.utils.data import download_file

import math
from PIL import Image
import vg
import glob
from alive_progress import alive_bar



import numpy as np
#вдохновляемся добрыми авторами в интернете для покраски звёзд
redco = [ 1.62098281e-82, -5.03110845e-77, 6.66758278e-72, -4.71441850e-67, 1.66429493e-62, -1.50701672e-59, -2.42533006e-53, 8.42586475e-49, 7.94816523e-45, -1.68655179e-39, 7.25404556e-35, -1.85559350e-30, 3.23793430e-26, -4.00670131e-22, 3.53445102e-18, -2.19200432e-14, 9.27939743e-11, -2.56131914e-07,  4.29917840e-04, -3.88866019e-01, 3.97307766e+02]
greenco = [ 1.21775217e-82, -3.79265302e-77, 5.04300808e-72, -3.57741292e-67, 1.26763387e-62, -1.28724846e-59, -1.84618419e-53, 6.43113038e-49, 6.05135293e-45, -1.28642374e-39, 5.52273817e-35, -1.40682723e-30, 2.43659251e-26, -2.97762151e-22, 2.57295370e-18, -1.54137817e-14, 6.14141996e-11, -1.50922703e-07,  1.90667190e-04, -1.23973583e-02,-1.33464366e+01]
blueco = [ 2.17374683e-82, -6.82574350e-77, 9.17262316e-72, -6.60390151e-67, 2.40324203e-62, -5.77694976e-59, -3.42234361e-53, 1.26662864e-48, 8.75794575e-45, -2.45089758e-39, 1.10698770e-34, -2.95752654e-30, 5.41656027e-26, -7.10396545e-22, 6.74083578e-18, -4.59335728e-14, 2.20051751e-10, -7.14068799e-07,  1.46622559e-03, -1.60740964e+00, 6.85200095e+02]

redco = np.poly1d(redco)
greenco = np.poly1d(greenco)
blueco = np.poly1d(blueco)

def temp2rgb(temp):

    red = redco(temp)
    green = greenco(temp)
    blue = blueco(temp)

    if red > 255:
        red = 255
    elif red < 0:
        red = 0
    if green > 255:
        green = 255
    elif green < 0:
        green = 0
    if blue > 255:
        blue = 255
    elif blue < 0:
        blue = 0

    color = [red/255,
             green/255,
             blue/255]
    return color

hard = 100 #easy
rsky = 500
sise = 0.5

k57 = 206265/3600
df = pd.read_csv('ZVZDA.csv')
h = df['h'] #прямое восхождение
m = df['m']
s = df['s']
dd = df['dd']
dm = df['dm']
ds = df['ds']
znak = df['znak']

mzv = df['mV']
BV = df['B-V']

dfn = pd.read_csv('NietMolotov.csv')
hn = dfn['h'] #прямое восхождение
mn = dfn['m']

ddn = dfn['dd']
dmn = dfn['dm']

N = dfn['N']
namen = dfn['name']

ra2 = ['']*len(h)
dec2 = ['']*len(h)

ra2n = ['']*len(hn)
dec2n = ['']*len(hn)

#загружаем названия картинок в один текстовый файл
# na = ['']*10803
# jais = 0
# for i in range(-176,184):
#     for j in range(-14,16):
#         namex = 'waypics/'+str(i)+'!'+str(j)+'!'+'*'
#         namexx = glob.glob(namex) #поиск картинки по первым буквам
#         na[jais] = namexx[0]
#         jais += 1
#
#     print(i)
# my_df = pd.DataFrame(na, columns=['col1'])
# my_df.to_csv('names.csv')

#location пришлось перенести наверх
fi = np.random.randint(-90, high=90, size=None, dtype=int)
lmbdaaa = np.random.randint(-180, high=180, size=None, dtype=int)
loc = EarthLocation.from_geodetic(
    lon=lmbdaaa*u.deg, lat=fi*u.deg)

#time
tx = np.random.randint(0, high=1440, size=None, dtype=int)
if len(str(tx%60))==2:
    tx = str(tx//60)+':'+str(tx%60//1)
else:
    tx = str(tx//60)+':'+'0'+str(tx%60//1)
observing_date = Time('2020-12-18 '+tx)

altaz = AltAz(location=loc, obstime=observing_date)


sky = [['name','x1','y1','x2','y2'] for _ in range(10800)]
d22f = pd.read_csv('names.csv')
nams = d22f['col1']
k = 0
skyl = [['',''] for _ in range(360)]
for i in range(10800):
    sky[i][0] = nams[i]

print('Взлом баз данных NASA для создания изображений Млечного пути')
with alive_bar(360, force_tty=True) as bar:
    for i in range(-176, 184): #-176 184
         for b in range(-14,16):
            if i < 0:
                l = 360+i
            else:
                l = i
            part = SkyCoord(l * u.degree, b * u.degree, frame='galactic')
            part2 = SkyCoord(l * u.degree, (b-1) * u.degree, frame='galactic')

            oc_altazg1 = part.transform_to(altaz)
            oc_altazg2 = part2.transform_to(altaz)




            r = rsky * math.tan((math.pi / 2 - oc_altazg1.alt.degree / k57) / 2)
            sky[k][1] = 500+math.sin(oc_altazg1.az.degree / k57) * r
            sky[k][2] = 500-math.cos(oc_altazg1.az.degree / k57) * r






            r = rsky * math.tan((math.pi / 2 - oc_altazg2.alt.degree / k57) / 2)
            sky[k][3] = 500 + math.sin(oc_altazg2.az.degree / k57) * r
            sky[k][4] = 500 - math.cos(oc_altazg2.az.degree / k57) * r





            k += 1
         bar()




print('Анализ данных...')
for i in range(len(namen)):
    namen[i] = " ".join(namen[i].split())
for i in range(len(h)):
    ra2[i] = float(h[i])*15+float(m[i])/4+float(s[i])/4/60
    if znak[i] == '+':
        dec2[i] = float(dd[i])+float(dm[i])/60+float(ds[i])/3600
    else:
        dec2[i] = -float(dd[i])-float(dm[i])/60-float(ds[i])/3600

for i in range(len(hn)):
    ra2n[i] = float(hn[i])*15+float(mn[i])/4
    if float(ddn[i]) > 0:
        dec2n[i] = float(ddn[i])+float(dmn[i])/60
    else:
        dec2n[i] = float(ddn[i])-float(dmn[i])/60

c = SkyCoord(ra2*u.degree,dec2*u.degree, frame = 'icrs')

cn = SkyCoord(ra2n*u.degree,dec2n*u.degree, frame = 'icrs')




oc_altaz = c.transform_to(altaz)
oc_altazn = cn.transform_to(altaz)

sky_coords = []
for i in range(len(oc_altaz)):
    sky_coords.append([oc_altaz[i].az.degree,oc_altaz[i].alt.degree])

sky_coordsn = []
for i in range(len(oc_altazn)):
    sky_coordsn.append([oc_altazn[i].az.degree,oc_altazn[i].alt.degree])


nab = [['x','y','B-V','mV'] for _ in range(len(ra2))]
nabn = [['x','y','N','name'] for _ in range(len(ra2))]

Mwas = [0] * 110

for i in range(len(ra2)):
    r = rsky*math.tan((math.pi/2-sky_coords[i][1]/k57)/2)
    nab[i][0] = math.cos(sky_coords[i][0]/k57)*r
    nab[i][1] = math.sin(sky_coords[i][0]/k57)*r
    nab[i][2] = BV[i]
    nab[i][3] = mzv[i]

for i in range(len(ra2n)):
    r = rsky*math.tan((math.pi/2-sky_coordsn[i][1]/k57)/2)
    if r>rsky:
        Mwas[i] = 1
    nabn[i][0] = math.cos(sky_coordsn[i][0]/k57)*r
    nabn[i][1] = math.sin(sky_coordsn[i][0]/k57)*r
    nabn[i][2] = N[i]
    nabn[i][3] = namen[i]

print('Текущая широта',fi,'Нажмите любую клавишу для загрузки интерфейса (больше ничего не нажимайте, оно точно работает)')

Messier=1
mk = 0

class MyPaintWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)


    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        global Mx,My,Messier,Mwas
        global nab,ra2,sky


        for i in range(k):
            if ((sky[i][3]-500)**2 + (sky[i][4]-500)**2)**0.5 < 500:
                with self.canvas:


                    vec1 = np.array([0,1,0])
                    vec2 = np.array([ sky[i][1]-sky[i][3], sky[i][2]-sky[i][4],0])
                    if sky[i][1]>sky[i][3]:
                        ang = 360 - vg.angle(vec1, vec2)
                    else:
                        ang = vg.angle(vec1, vec2)
                    kivy.graphics.PushMatrix()
                    self.rotation = Rotate(origin=(sky[i][3], sky[i][4]),angle=ang)
                    self.bind(center=lambda _, value: setattr(self.rotation, "origin", value))

                    im = Image.open(sky[i][0])
                    width, height = im.size
                    A = sky[i][0].split("!")
                    if float(A[1])<0:
                        A[1] = 1-float(A[1])
                    Color(1, 1, 1, (16-abs(float(A[1])))/15)
                    dlina = ((sky[i][1]-sky[i][3])**2 + (sky[i][2]-sky[i][4])**2)**0.5
                    Rectangle(source=sky[i][0],pos=(sky[i][3], sky[i][4]), size=(dlina*width/height, dlina))
                    PopMatrix()



        # with self.canvas: #стрелочка
        #     kivy.graphics.PushMatrix()
        #     self.rotation = Rotate(origin=(300, 300),angle=90)
        #     self.bind(center=lambda _, value: setattr(self.rotation, "origin", value))
        #     Color(1, 1, 1, 1)
        #     im = Image.open('s.png')
        #     width, height = im.size
        #     d = 700
        #     Rectangle(source='s.png',pos=(300, 300), size=(width/height*750, 750))
        #     PopMatrix()

        #обновляем звёздочки, фигачим мессьешку

        for i in range(len(ra2)):
            if (nab[i][0]**2 + nab[i][1]**2)**0.5 < 500:
                with self.canvas:
                    ttt = 4600 * (1 / (0.92 * nab[i][2] + 1.7) + 1 / (0.92 * nab[i][2] + 0.62))
                    rgbb = temp2rgb(ttt)

                    Color(rgbb[0], rgbb[1], rgbb[2],1)#цвет

                    d = sise*(10**((6.5-nab[i][3])/2.5))**0.5
                    if nab[i][3]<0.7:
                        d = sise * (10 ** ((6.5 - nab[i][3]) / 2.5)) ** 0.5 / (sise*(10**((6.5-0.7)/2.5))**0.5) * 0.1 +(sise*(10**((6.5-0.7)/2.5))**0.5)
                    Ellipse(pos=(500 + nab[i][1], 500 - nab[i][0]), size=(d, d))

        for i in range(1050):
            Messier = np.random.randint(0, high=109, size=None, dtype=int)
            if Mwas[Messier] != 1:
                break
        Mwas[Messier] = 1
        Mx = 500 + nabn[Messier][1]
        My = 500 - nabn[Messier][0]

        with self.canvas:
            Color(0.1, 0.1, 0.1, 1)  # цвет

            d = 100
            Rectangle( pos=(995, 0), size=(5, 1048))

        #Текст MX
        my_label = CoreLabel()
        my_label.text = "M"+str(Messier+1)
        my_label.refresh()
        text = my_label.texture

        with self.canvas:
            Color(1, 1, 1, 1)  # цвет

            d = 100
            Rectangle( pos=(1050, 900), size=(50*len(str(Messier+1)), 50), texture=text)

        my_label = CoreLabel()
        my_label.text = nabn[Messier][3]
        my_label.refresh()
        text = my_label.texture


        with self.canvas:
            Color(1, 1, 1, 1)  # цвет

            d = 100
            Rectangle( pos=(1050, 800), size=(20*len(nabn[Messier][3]), 30), texture=text)

        with self.canvas:
            Color(1, 1, 1, 1)
            ky = 'M' + str(Messier+1) + '.jpg'
            im = Image.open(ky)
            width, height = im.size
            d = 700
            Rectangle(source=ky,pos=(1050, 50), size=(width/height*750, 750))


        # for i in range(len(ra2n)):
        #     if (nabn[i][0]**2 + nabn[i][1]**2)**0.5 < 500:
        #         with self.canvas:
        #             ky = 'M' + str(i+1) + '.jpg'
        #
        #             d = 15
        #             Rectangle(source=ky,pos=(500 + nabn[i][1], 500 - nabn[i][0]), size=(d, d))
#      Все объекты мессье на фото


    def on_touch_down(self, touch):
        global Mx,My,hard,Messier,mk
        print(touch,self)
        #считаем расстояние до мессьешки, рисуем мессьешку, ставим крест с цветом в зависимости от правильности, делаем новую мессьешку
        g = ((Mx - touch.x) ** 2 + (My - touch.y) ** 2)/hard**2
        print(g)
        with self.canvas:
            Color(g, 1-g, 0, 1)
            d = 8.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, 1))
        with self.canvas:
            Color(g, 1-g, 0, 1)
            d = 8.
            Ellipse(pos=(touch.x, touch.y - d), size=(1, d))
        with self.canvas:
                 ky = 'M' + str(Messier+1) + '.jpg'

                 d = 15
                 Rectangle(source=ky,pos=(Mx, My), size=(d, d))


        for i in range(1000):
            Messier = np.random.randint(0, high=109, size=None, dtype=int)
            if Mwas[Messier] != 1:
                break
            if i == 999:
                print("Мессье здесь больше нет!") # (или вам не ОЧЕНЬ не повезло)
        print(Messier)
        Mwas[Messier] = 1
        Mx = 500 + nabn[Messier][1]
        My = 500 - nabn[Messier][0]

        my_label = CoreLabel()
        my_label.text = "M"+str(Messier+1)
        my_label.refresh()
        text = my_label.texture

        with self.canvas:
            Color(0, 0, 0, 1)  # цвет

            d = 100
            Rectangle( pos=(1050, 900), size=(300, 50))


        with self.canvas:
            Color(1, 1, 1, 1)  # цвет

            d = 100
            Rectangle( pos=(1050, 900), size=(50*len(str(Messier+1)), 50), texture=text)


        my_label = CoreLabel()
        my_label.text = nabn[Messier][3]
        my_label.refresh()
        text = my_label.texture


        with self.canvas:
            Color(0, 0, 0, 1)  # цвет

            d = 100
            Rectangle( pos=(1050, 800), size=(1000, 30))
        with self.canvas:
            Color(1, 1, 1, 1)  # цвет

            d = 100
            Rectangle( pos=(1050, 800), size=(20*len(nabn[Messier][3]), 30), texture=text)

        with self.canvas:
            Color(0, 0, 0, 1)

            Rectangle(source=ky,pos=(1050, 50), size=(1000, 750))

        # картинка
        with self.canvas:
            Color(1, 1, 1, 1)
            ky = 'M' + str(Messier+1) + '.jpg'
            im = Image.open(ky)
            width, height = im.size
            d = 700
            Rectangle(source=ky,pos=(1050, 50), size=(width/height*750, 750))





class NablaApp(App):
    def build(self):
        return MyPaintWidget()


if __name__ == '__main__':
    NablaApp().run()

