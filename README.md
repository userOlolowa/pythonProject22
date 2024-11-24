Смысл программы: отмечать объекты Мессье на скайчарте(а ещё, видимо, учить их финские названия).
![image](https://github.com/user-attachments/assets/50775e85-d4d1-497d-9a2c-58765bc80a56)


Как поставить на пайчарм: в верхней панели Git - clone - вставляем https://github.com/userOlolowa/pythonProject22 - проект открыт!

Первоначальная настройка:
1. Установите библиотеки, указанные в venv./requirments.txt, с помощью   pip install -r requirements.txt
2. Программа предназначена под разрешение экрана 2048*1024. Можно больше, но если меньше - картинка обрезается.
3. Запускать нужно файл /venv./Skripts/main.py. Все дальнейшие действия происходят в нём.
4. Скайчарт можно немного настроить под себя, в строке 205 можно указать пороговую точность, с которой нужно указать объект мессье, в 207 можно поменять размеры звёзд(если вдруг хотите понаблюдать в условиях Московского неба, поставьте sise = 0) 
5. Если вдруг у вас миллиард времени(на генерацию уходит 4 минуты), и вам хочется красивый Млечный путь - больше ничего не трогайте.
6. Если же вы хотите, чтобы скайчарт генерировался 10 секунд (это на моём phenom II 2013 года так, так оно побыстрее будет), закомментите 269 - 312 and 395 - 418 участки кода.

Сама программа:
1. 3апустите, как появится сообщение "нажмите клавишу" - нажимайте любую клавишу клавиатуры(желательно букву) держа чёрное окно открытым, после этого начнётся прогрузка графического интерфейса. До того, как всё прогрузится, НАЖАТИЕ ЧЕГО-ЛИБО НА МЫШКЕ УНИЧТОЖАЕТ ПРОГРАММУ.
2. После прогрузки интерфейса, тыкайте на местоположение мессьешки на скайчарте, после ЛКМ/ПКМ/СКМ/и т.д. (любая кнопка на мышке) ваш ответ отметится крестиком, и появится реальное местоположение этого объекта.
