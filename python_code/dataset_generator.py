import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from cross_section_sphere import CrossSection


engine = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)

coon = engine.connect()
sql_query = """CREATE TABLE DATA4SIMPLE_SPHERE
(ID INT NOT NULL AUTO_INCREMENT,
R TEXT,
Material TEXT,
PRIMARY KEY (ID))"""
coon.execute(sql_query)

columns = [f'{i}nm DOUBLE' for i in np.arange(300, 1200, 2)]
columns_str = ', '.join(columns)
sql_query = f"""CREATE TABLE CS4SIMPLE_SPHERE
(ID INT NOT NULL AUTO_INCREMENT,
DATA_ID INT,
{columns_str},
PRIMARY KEY (ID),
FOREIGN KEY (DATA_ID) REFERENCES DATA4SIMPLE_SPHERE(ID))"""
coon.execute(sql_query)
coon.close()


material = ["Ag", "Au", "Cu"]
columns = [f'{i}nm' for i in np.arange(300, 1200, 2)]

with engine.begin() as conn:
    for _ in range(3000):
        r = np.array(sorted([np.random.randint(30, 201) for _ in range(2)]))
        eps = np.random.choice(material, 2, replace=False).tolist()
        srt_r = '|'.join([str(i) for i in r])
        srt_eps = '|'.join([str(i) for i in eps])

        result = conn.execute(f"""SELECT ID FROM DATA4SIMPLE_SPHERE WHERE R={srt_r}
                                AND Material= '{srt_eps}'""")
        existing_id = result.scalar()

        if existing_id:
            print(f"Запис з R = {srt_r} и "
                  f"Material = {srt_eps} вже є в таблиці")
            continue

        cs = CrossSection(r * 10e-9, eps, L=0)
        cs.calc_cross_section()
        # вставка данных в таблицу DATA4SIMPLE_SPHERE и получение ID
        result = conn.execute(f"""INSERT INTO DATA4SIMPLE_SPHERE (R, Material) VALUES ('{srt_r}', 
                              '{srt_eps}')""")
        data_id = result.lastrowid

        # вставка данных в таблицу CS4SIMPLE_SPHERE
        values = ', '.join(str(i) for i in cs.Cross_section)
        conn.execute(f"INSERT INTO CS4SIMPLE_SPHERE (DATA_ID, {', '.join(columns)}) VALUES ({data_id}, {values})")

