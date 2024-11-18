# import requests
#
# for i in range(1,110):
#     k = 'https://kosmoved.ru/Foto_DipSky/M'+str(i)+'.jpg'
#     ky = 'M'+str(i)+'.jpg'
#     img_data = requests.get(k).content
#     with open(ky, 'wb') as handler:
#         handler.write(img_data)

from PIL import Image
import numpy as np

way = Image.open('way.jpg')

# iio.imwrite('green.png', B3vis)
way = np.array(way)

# print(way.shape)
# print(way)
# print(way.mode)
print(way[0][0])
for i in range(0,360):
    for j in range(0,30): #30

        x1 = int((1463+(j-15)*16.66)//1)
        x2 = int((1463+(j-15+1)*16.66)//1)
        x3 = int((i*16.66)//1)
        x4 = int(((i+1)*16.66)//1)
        sps = np.array([[[5,5,5] for _ in range(x4-x3)] for kjidasjof in range(x2-x1)])
        # sps = np.zeros((x2-x1, x4-x3, 3))
        for xi in range(x1,x2):
            for xj in range(x3,x4):
                for z in range(3):
                    sps[xi-x1][xj-x3][z] = way[xi][xj][z]

        name = str(183-i)+'!'+str(15-j)+'!'+str(x2-x1)+'!'+str(x4-x3)+'.jpg'
        print(sps[0][0]-way[0][0])
        # print(sps.shape)
        # print(sps)
        im = Image.fromarray((sps).astype(np.uint8))

        # смотрим что получилось
        im.save(name)
