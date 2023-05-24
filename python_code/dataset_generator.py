import numpy as np
from sqlalchemy import create_engine, text as sql_text
from cross_section_sphere import CrossSection


tabl_data_name = 'DATA4DOUBLE_SPHERE_WITH_EPS_OUT'
tabl_cs_name = 'CS4DOUBLE_SPHERE_WITH_EPS_OUT'

engine = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)

coon = engine.connect()
sql_query = sql_text(f"""CREATE TABLE {tabl_data_name}
(ID INT NOT NULL AUTO_INCREMENT,
R TEXT,
Material TEXT,
PRIMARY KEY (ID))""")
coon.execute(sql_query)

columns = [f'{i}nm DOUBLE' for i in np.arange(300, 1202, 2)]
columns_str = ', '.join(columns)
sql_query = sql_text(f"""CREATE TABLE {tabl_cs_name}
(ID INT NOT NULL AUTO_INCREMENT,
DATA_ID INT,
{columns_str},
PRIMARY KEY (ID),
FOREIGN KEY (DATA_ID) REFERENCES {tabl_data_name}(ID))""")
coon.execute(sql_query)
coon.close()


material = ["Cu", "Au"]
columns = [f'{i}nm' for i in np.arange(300, 1202, 2)]
size = 2
count = 3000

for i in range(count):
    print(f'\r{i}/{count} ', end='')
    conn = engine.connect()
    r = np.array(sorted(np.random.randint(30, 201, size=size)))
    eps = [np.random.choice(material)]
    while len(eps) < size:
        a = np.random.choice(material)
        if eps[-1] != a:
            eps.append(a)

    srt_r = '|'.join([str(i) for i in r])
    srt_eps = '|'.join([str(i) for i in eps])

    result = conn.execute(sql_text(f"""SELECT ID FROM {tabl_data_name} WHERE R='{srt_r}' AND Material='{srt_eps}' """))
    existing_id = result.scalar()

    if existing_id:
        print(f"Запис з R = {srt_r} и "
              f"Material = {srt_eps} вже є в таблиці")
        continue

    cs = CrossSection(r * 10e-9, eps, L=0)
    cs.calc_cross_section()
    # вставка данных в таблицу DATA4SIMPLE_SPHERE и получение ID

    result = conn.execute(sql_text(f"""INSERT INTO {tabl_data_name} (R, Material) VALUES ('{srt_r}', '{srt_eps}')"""))
    data_id = result.lastrowid
    # вставка данных в таблицу CS4SIMPLE_SPHERE
    values = ', '.join(str(i) for i in cs.Cross_section)
    conn.execute(sql_text(f"INSERT INTO {tabl_cs_name} (DATA_ID, {', '.join(columns)}) VALUES ({data_id}, {values})"))
    conn.close()
