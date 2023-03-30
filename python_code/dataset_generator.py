import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from cross_section_sphere import CrossSection


engine = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)

# coon = engine.connect()
# sql_query = """CREATE TABLE DATA4SIMPLE_SPHERE
# (ID INT NOT NULL AUTO_INCREMENT,
# R DOUBLE,
# Material DOUBLE,
# PRIMARY KEY (ID))"""
# coon.execute(sql_query)
#
# columns = [f'{i}nm DOUBLE' for i in np.arange(100, 1000, 1)]
# columns_str = ', '.join(columns)
# sql_query = f"""CREATE TABLE CS4SIMPLE_SPHERE
# (ID INT NOT NULL AUTO_INCREMENT,
# DATA_ID INT,
# {columns_str},
# PRIMARY KEY (ID),
# FOREIGN KEY (DATA_ID) REFERENCES DATA4SIMPLE_SPHERE(ID))"""
# coon.execute(sql_query)
# coon.close()


material = [2.3, 1.7, 2.9, 1.6]
columns = [f'{i}nm' for i in np.arange(100, 1000, 1)]

with engine.begin() as conn:
    for _ in range(3000):
        r = np.random.randint(60, 250) + round(np.random.rand(), 2)
        m = material[np.random.randint(0, len(material))]
        cs = CrossSection([r*10e-9], [m], L=0)

        result = conn.execute(f"SELECT ID FROM DATA4SIMPLE_SPHERE WHERE R={r} AND Material={m}")
        existing_id = result.scalar()

        if existing_id:
            print(f"Запись с R={r} и Material={m} уже есть в таблице")
            continue

        cs.calc_cross_section()
        # вставка данных в таблицу DATA4SIMPLE_SPHERE и получение ID
        result = conn.execute(f"INSERT INTO DATA4SIMPLE_SPHERE (R, Material) VALUES ({r}, {m})")
        data_id = result.lastrowid

        # вставка данных в таблицу CS4SIMPLE_SPHERE
        values = ', '.join(str(i) for i in cs.Cross_section)
        conn.execute(f"INSERT INTO CS4SIMPLE_SPHERE (DATA_ID, {', '.join(columns)}) VALUES ({data_id}, {values})")








