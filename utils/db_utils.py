import os
import psycopg
import subprocess
import pandas as pd

# 连接配置
conn_info = "host=localhost user=postgres password=123456"
host = 'localhost'
port = '5432'
user = 'postgres'
password = '123456'

database_path = "/home/zhanghang/opt/projects/researchprojects/text2PLSQL/sftDataProcessing/datasets/spider_data/database"
database_names = sorted(os.listdir(database_path))

input_path = '/home/zhanghang/opt/projects/researchprojects/text2PLSQL/sftDataProcessing/datasets/spider_pg_dump/'

def get_tables_info(database_name):

    conn_db_info = f"""host=localhost dbname={database_name} user=postgres password=123456"""

    tables_info = {}
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT tablename
                            FROM pg_catalog.pg_tables
                            WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';""")
            result = cur.fetchall()
            table_names = [table_name[0] for table_name in result]

            for table_name in table_names:
                cur.execute(f"""SELECT column_name, data_type
                                FROM information_schema.columns
                                WHERE table_name = '{table_name}';""")
                result = cur.fetchall()
                tables_info[table_name] = [item for item in result]
                
    return tables_info

# 删除再创建数据库的函数
def recreate_databases(conn_info, databases):
    with psycopg.connect(conn_info) as conn:
        conn.autocommit = True  # 必须启用自动提交，以允许删除和创建数据库
        with conn.cursor() as cur:
            for db_name in databases:
                # 尝试删除数据库（如果存在）
                cur.execute(f"""SELECT pg_terminate_backend(pg_stat_activity.pid)
                                FROM pg_stat_activity
                                WHERE pg_stat_activity.datname = '{db_name}' AND
                                        pid <> pg_backend_pid();""")
                cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
                # 创建数据库
                cur.execute(f"CREATE DATABASE {db_name};")

def import_database(host, port, user, password, dbname, input_file):
    command = f"psql -h {host} -p {port} -U {user} -d {dbname} -f {input_file}"
    env = {**os.environ, "PGPASSWORD": password}
    
    process = subprocess.run(command, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def restore_databases(conn_info, host, port, user, password, database_names):
    try:
        # 删除并重建数据库
        recreate_databases(conn_info, database_names)

        # 导入数据
        for dbname in database_names:
            input_file = f"""{input_path}{dbname}.sql"""
            import_database(host, port, user, password, dbname.lower(), input_file)
    except Exception as e:
        print(f"Error restoring databases {database_names}: {e}")

def execute_sql(database_conn_info, sql):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

def check_plsql_executability(generated_plsql, call_plsqls, database_name):
    execution_error = None
    try:
        database_conn_info = f"""host=localhost user=postgres password=123456 dbname={database_name}"""
        restore_databases(conn_info, host, port, user, password, [database_name])
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute(generated_plsql)
                for call in call_plsqls:
                    cur.execute(call)
    except Exception as e:
        execution_error = str(e)
    
    return execution_error

def fetch_query_results(database_conn_info, query):
    with psycopg.connect(database_conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result

def compare_plsql(database_name, tables, plsql1, plsql2, call_plsqls):
    try:
        database_conn_info = f"""host=localhost user=postgres password=123456 dbname={database_name}"""
        results1 = []
        results2 = []

        restore_databases(conn_info, host, port, user, password, [database_name])
        
        execute_sql(database_conn_info, plsql1)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        for table in tables:
            select_query = f"""select * from \"{table}\";"""
            results1.append(fetch_query_results(database_conn_info, select_query))
            results1 = [pd.DataFrame(result) for result in results1]
        
        restore_databases(conn_info, host, port, user, password, [database_name])
        
        execute_sql(database_conn_info, plsql2)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        for table in tables:
            select_query = f"""select * from \"{table}\";"""
            results2.append(fetch_query_results(database_conn_info, select_query))
            results2 = [pd.DataFrame(result) for result in results2]

        same = True

        for (df1, df2) in zip(results1, results2):
            if not df1.equals(df2):
                same = False
                break

        return same
    
    except Exception as e:
        print(e)
        return False
    
def get_db_schema_sequence(schema):
    schema_sequence = "database schema :\n"
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        
        # if table_comment != "":
        #     table_name += " ( comment : " + table_comment + " )"

        column_info_list = []
        for column_name, column_type, column_comment, column_content, pk_indicator in \
            zip(table["column_names"], table["column_types"], table["column_comments"], table["column_contents"], table["pk_indicators"]):
            
            additional_column_info = []
            # column type
            additional_column_info.append(column_type)
            # pk indicator
            if pk_indicator != 0:
                additional_column_info.append("primary key")
            # column comment
            if column_comment != "":
                additional_column_info.append("comment : " + column_comment)
            # representive column values
            if len(column_content) != 0:
                additional_column_info.append("values : " + " , ".join(map(str, column_content)))
            
            column_info_list.append(column_name + " ( " + " | ".join(additional_column_info) + " )")
        
        schema_sequence += "table "+ table_name + " , columns = [ " + " , ".join(column_info_list) + " ]\n"

    if len(schema["foreign_keys"]) != 0:
        schema_sequence += "foreign keys :\n"
        for foreign_key in schema["foreign_keys"]:
            schema_sequence += "{}.{} = {}.{}\n".format(foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3])
    else:
        schema_sequence += "foreign keys : None\n"

    return schema_sequence.strip()
