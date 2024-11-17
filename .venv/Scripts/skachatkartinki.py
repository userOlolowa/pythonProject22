import requests

for i in range(1,110):
    k = 'https://kosmoved.ru/Foto_DipSky/M'+str(i)+'.jpg'
    ky = 'M'+str(i)+'.jpg'
    img_data = requests.get(k).content
    with open(ky, 'wb') as handler:
        handler.write(img_data)

