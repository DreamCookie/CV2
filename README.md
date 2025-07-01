# CV-Detector
## Оглавление

- [Особенности](#особенности)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Использование](#использование)
  - [Обработка изображений из папки](#обработка-изображений-из-папки)
  - [Камера](#камера)
  - [YOLO](#yolo)
- [Модули](#модули)

## Особенности

- **Простой классический детектор**  - пороговая бинаризация, поиск контуров, фильтрация по площади
- **Опциональный NN-детектор** с использованием `ultralytics` (YOLOv8)


## Структура проекта

## Структура проекта

```text
CV/
├── .venv/                     
├── config/                    
│   ├── detection.yaml         
│   └── req.txt                
├── detector/                  
│   ├── base.py                
│   ├── contour_detector.py    
│   ├── nn_detector.py         
│   ├── adaptive_background.py
│   ├── calibration.py              
│   ├── coord_transform.py   
│   ├── mask.py 
│   ├── ros_coord_transform.py  
│   └── preprocessing.py         
├── models/                    
│   └── yolo.pt                
├── test_img/                  
│   ├── 1.jpg                  
│   ├── 2.jpg                  
│   ├── 3.jpg                  
│   └── 4.jpg                  
├── main.py                    
├── utils.py                   
└── README.md                  
```


## Требования

- Python 3.7+
- OpenCV
- NumPy
- PyYAML
- ultralytics

Список зависимостей приведён в req.txt.


## Использование

### команды
```bash
 python src/main.py --input-dir test_img/
```

```bash
python src/main.py --input-dir test_img/ --use-nn
```

```bash
python src/main.py --input-dir test_img/ --use-ros
```

```bash
python src/main.py --input-dir test_img/ --use-nn --use-ros
```

```bash
python src/main.py --input-dir test_img/ --mode tools
python src/main.py --input-dir test_img/ --mode workpieces
```

```bash
python main.py --camera 0
```

```bash
python main.py --camera 0 --use-nn
```

```bash
python src/main.py --camera 0 --use-ros
```
```bash
python src/main.py --camera 0 --use-nn --use-ros
```

### Модули

detector/contour_detector.py — классический алгоритм поиска контуров

detector/nn_detector.py — детектор на базе ultralytics.YOLO

detector/base.py — интерфейс Detector, класс BoundingBox


## в планах
1. Калибровка камеры и перевод пикселей в мм.

2. Интеграция с роботом через ROS2/Modbus TCP.

3. ??? Автоматическая генерация синтетического датасета. (может остановлюсь на классическом алгоритме с бинаризацией)

4. ??? Дообучение модели YOLO на собственных заготовках.



