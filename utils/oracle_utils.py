import os
import sys
import oracledb
from pathlib import Path
import configparser
import re

sql_dir_path = "/home/zhanghang/opt/projects/researchprojects/text2PLSQL/oracle/databases/"

class OracleConnectionManager:
    """
    Oracle连接管理器，支持上下文管理和连接复用
    """
    def __init__(self, username="system", password="MyPassword123", 
                 host="localhost", port="1521", service_name="XEPDB1"):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.importer = None
    
    def __enter__(self):
        """进入上下文时建立连接"""
        self.importer = OracleSchemaImporter(
            self.username, self.password, self.host, self.port, self.service_name
        )
        if not self.importer.connect():
            raise ConnectionError("无法连接到Oracle数据库")
        return self.importer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时断开连接"""
        if self.importer:
            self.importer.disconnect()
            self.importer = None

class OracleSchemaImporter:
    def __init__(self, username, password, host, port, service_name):
        """
        初始化Oracle连接参数
        """
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.connection = None
        
    def connect(self):
        """
        连接到Oracle数据库（需要有创建用户的权限）
        """
        try:
            # 构建连接字符串
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            self.connection = oracledb.connect(
                user=self.username, 
                password=self.password, 
                dsn=dsn
            )
            return True
        except oracledb.Error as e:
            print(f"连接Oracle数据库失败: {e}")
            return False
    
    def disconnect(self):
        """
        断开数据库连接
        """
        if self.connection:
            self.connection.close()
    
    def extract_schema_name_from_filename(self, filename):
        """
        从文件名提取Schema名
        例: allergy_1.sqlite.sql -> ALLERGY_1
        """
        # 移除所有后缀
        schema_name = filename
        for suffix in ['.sqlite.sql', '.sql']:
            if schema_name.endswith(suffix):
                schema_name = schema_name[:-len(suffix)]
                break
        
        # 确保名称符合Oracle标识符规范
        schema_name = re.sub(r'[^a-zA-Z0-9_]', '_', schema_name)
        
        # Oracle用户名限制
        if len(schema_name) > 30:
            schema_name = schema_name[:30]
        
        return schema_name.lower()
    
    def create_schema(self, schema_name):
        """
        创建Oracle Schema（用户）
        """
        try:
            cursor = self.connection.cursor()
            
            # 生成密码
            password = f"{schema_name.lower()}_pwd"
            
            # 检查用户是否已存在
            cursor.execute("""
                SELECT COUNT(*) FROM all_users WHERE username = :username
            """, [schema_name])
            
            user_exists = cursor.fetchone()[0] > 0
            
            if user_exists:
                # 删除现有用户
                try:
                    cursor.execute(f"DROP USER {schema_name} CASCADE")
                except oracledb.Error as e:
                    if "does not exist" not in str(e).lower():
                        print(f"删除用户失败: {e}")
            
            # 创建用户
            cursor.execute(f"""
                CREATE USER "{schema_name}" IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            
            # 授权
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
            
            # 提交
            self.connection.commit()
            cursor.close()
            
            return schema_name, password
            
        except oracledb.Error as e:
            print(f"创建Schema失败: {e}")
            return None, None
    
    def connect_to_schema(self, schema_name, password):
        """
        连接到指定Schema
        """
        try:
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            schema_connection = oracledb.connect(
                user=schema_name, 
                password=password, 
                dsn=dsn
            )
            return schema_connection
        except oracledb.Error as e:
            print(f"连接Schema失败: {e}")
            return None
    
    def execute_sql_file_in_schema(self, sql_file, schema_name, password):
        """
        在指定Schema中执行SQL文件
        """
        # 连接到指定Schema
        schema_connection = self.connect_to_schema(schema_name, password)
        if not schema_connection:
            return False
        
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句（以分号为分隔符）
            sql_statements = []
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    sql_statements.append(statement)
            
            if not sql_statements:
                print(f"SQL文件无效: 没有找到有效的SQL语句")
                schema_connection.close()
                return False
            
            cursor = schema_connection.cursor()
            success_count = 0
            
            for i, sql in enumerate(sql_statements):
                try:
                    cursor.execute(sql)
                    success_count += 1
                except oracledb.Error as e:
                    error_msg = str(e)
                    # 忽略常见的无害错误
                    if "table or view does not exist" in error_msg.lower() and "drop" in sql.lower():
                        continue
                    elif "name is already used by an existing object" in error_msg.lower():
                        # 提取表名
                        table_match = re.search(r'CREATE TABLE\s+"?(\w+)"?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            try:
                                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                                cursor.execute(sql)
                                success_count += 1
                            except Exception as drop_error:
                                print(f"重新创建表失败 {table_name}: {drop_error}")
                        continue
                    else:
                        print(f"SQL执行失败 [{i+1}]: {e}")
                        continue
            
            # 提交事务
            schema_connection.commit()
            cursor.close()
            schema_connection.close()
            
            return success_count > 0
            
        except Exception as e:
            print(f"执行SQL文件失败: {e}")
            if schema_connection:
                schema_connection.close()
            return False
    
    def recreate_schema_from_sql(self, schema_name, sql_file_path):
        """
        重新创建Oracle Schema - 先删除后创建
        
        Args:
            schema_name (str): Schema名称（用户名）
            sql_file_path (str): SQL文件路径
        
        Returns:
            tuple: (success, schema_name, password) 成功标志、schema名、密码
        """
        try:
            cursor = self.connection.cursor()
            
            # 生成密码
            password = f"{schema_name.lower()}_pwd"
            
            # 1. 检查并删除现有Schema
            cursor.execute("""
                SELECT COUNT(*) FROM all_users WHERE username = :username
            """, [schema_name])
            
            user_exists = cursor.fetchone()[0] > 0
            
            if user_exists:
                try:
                    cursor.execute(f"DROP USER {schema_name} CASCADE")
                except oracledb.Error as e:
                    if "does not exist" not in str(e).lower():
                        print(f"删除Schema失败: {e}")
                        return False, None, None
            
            # 2. 创建新的Schema
            cursor.execute(f"""
                CREATE USER "{schema_name}" IDENTIFIED BY {password}
                DEFAULT TABLESPACE USERS
                TEMPORARY TABLESPACE TEMP
            """)
            
            # 3. 授权
            cursor.execute(f"GRANT CONNECT, RESOURCE TO {schema_name}")
            cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {schema_name}")
            
            # 提交创建用户的操作
            self.connection.commit()
            cursor.close()
            
            # 4. 执行SQL文件导入数据
            if self._execute_sql_file_in_schema(sql_file_path, schema_name, password):
                return True, schema_name, password
            else:
                print(f"Schema数据导入失败")
                return False, schema_name, password
                
        except oracledb.Error as e:
            print(f"重新创建Schema失败: {e}")
            return False, None, None

    def _execute_sql_file_in_schema(self, sql_file_path, schema_name, password):
        """
        在指定Schema中执行SQL文件（内部方法）
        
        Args:
            sql_file_path (str): SQL文件路径
            schema_name (str): Schema名称
            password (str): Schema密码
        
        Returns:
            bool: 执行成功标志
        """
        # 连接到指定Schema
        schema_connection = self.connect_to_schema(schema_name, password)
        if not schema_connection:
            return False
        
        try:
            # 读取SQL文件
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句
            sql_statements = []
            statements = sql_content.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    sql_statements.append(statement)
            
            if not sql_statements:
                print(f"SQL文件无效: 没有找到有效的SQL语句")
                schema_connection.close()
                return False
            
            cursor = schema_connection.cursor()
            success_count = 0
            
            # 执行每条SQL语句
            for i, sql in enumerate(sql_statements):
                try:
                    cursor.execute(sql)
                    success_count += 1
                except oracledb.Error as e:
                    error_msg = str(e)
                    # 忽略常见的无害错误
                    if "table or view does not exist" in error_msg.lower() and "drop" in sql.lower():
                        continue
                    elif "name is already used by an existing object" in error_msg.lower():
                        # 提取表名并重建
                        table_match = re.search(r'CREATE TABLE\s+"?(\w+)"?', sql, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                            try:
                                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                                cursor.execute(sql)
                                success_count += 1
                            except Exception as drop_error:
                                print(f"重新创建表失败 {table_name}: {drop_error}")
                        continue
                    else:
                        print(f"SQL执行失败 [{i+1}]: {e}")
                        continue
            
            # 提交事务
            schema_connection.commit()
            cursor.close()
            schema_connection.close()
            
            return success_count > 0
            
        except Exception as e:
            print(f"执行SQL文件失败: {e}")
            if schema_connection:
                schema_connection.close()
            return False

    def import_file_as_schema(self, sql_file):
        """
        为单个SQL文件创建Schema并导入数据
        """
        sql_path = Path(sql_file)
        
        if not sql_path.exists():
            print(f"文件不存在: {sql_file}")
            return False, None, None
        
        # 从文件名提取Schema名
        schema_name = self.extract_schema_name_from_filename(sql_path.name)
        
        # 1. 创建Schema
        schema_name, password = self.create_schema(schema_name)
        if not schema_name:
            return False, None, None
        
        # 2. 在Schema中导入数据
        success = self.execute_sql_file_in_schema(sql_path, schema_name, password)
        
        if success:
            print(f"Schema创建成功: {schema_name}")
            return True, schema_name, password
        else:
            print(f"Schema导入失败: {schema_name}")
            return False, schema_name, password
    
    def import_directory_as_schemas(self, sql_directory):
        """
        为目录中的每个SQL文件创建独立的Schema
        """
        sql_path = Path(sql_directory)
        
        if not sql_path.exists():
            print(f"目录不存在: {sql_directory}")
            return False
        
        sql_files = list(sql_path.glob("*.sql"))
        
        if not sql_files:
            print(f"目录中没有找到SQL文件: {sql_directory}")
            return False
        
        success_count = 0
        created_schemas = []
        
        for sql_file in sorted(sql_files):
            schema_name = self.extract_schema_name_from_filename(sql_file.name)
            
            # 1. 创建Schema
            created_schema, password = self.create_schema(schema_name)
            if not created_schema:
                continue
            
            # 2. 在Schema中导入数据
            if self.execute_sql_file_in_schema(sql_file, created_schema, password):
                success_count += 1
                created_schemas.append({
                    'file': sql_file.name,
                    'schema': created_schema,
                    'password': password,
                    'connection_string': f"{created_schema}/{password}@{self.host}:{self.port}/{self.service_name}"
                })
        
        print(f"批量导入完成: {success_count}/{len(sql_files)} 成功")
        
        if created_schemas:
            for schema_info in created_schemas:
                print(f"Schema: {schema_info['schema']} | 连接: {schema_info['connection_string']}")
        
        return success_count == len(sql_files)

# 全局连接管理器
_global_importer = None

def get_oracle_importer():
    """
    获取全局Oracle连接实例，如果不存在则创建
    """
    global _global_importer
    if _global_importer is None:
        _global_importer = OracleSchemaImporter(
            username="system",
            password="MyPassword123", 
            host="localhost",
            port="1521",
            service_name="XEPDB1"
        )
        # 尝试连接
        if not _global_importer.connect():
            print("无法连接到Oracle数据库")
            _global_importer = None
            return None
    return _global_importer

def close_oracle_connection():
    """
    关闭全局Oracle连接
    """
    global _global_importer
    if _global_importer is not None:
        _global_importer.disconnect()
        _global_importer = None

def recreate_database(database_name):
    """
    重新创建数据库schema（优化版本 - 复用连接）
    
    Args:
        database_name (str): 数据库名称
    
    Returns:
        tuple: (success, schema_name, password)
    """
    database_name = database_name.upper()
    
    # 获取全局连接实例
    importer = get_oracle_importer()
    if importer is None:
        return False, None, None
    
    # 重新创建schema
    success, schema_name, password = importer.recreate_schema_from_sql(
        schema_name=database_name,
        sql_file_path=sql_dir_path + database_name.lower() + ".sql"
    )
    
    if success:
        print(f"Schema重建成功: {schema_name}")
    else:
        print("Schema重建失败!")
    
    return success, schema_name, password

def recreate_databases(database_names):
    """
    批量重新创建多个数据库schema（复用同一个连接）
    
    Args:
        database_names (list): 数据库名称列表
    
    Returns:
        dict: {database_name: (success, schema_name, password)}
    """
    results = {}
    
    # 获取全局连接实例
    importer = get_oracle_importer()
    if importer is None:
        return {name: (False, None, None) for name in database_names}
    
    for database_name in database_names:
        database_name = database_name.upper()
        
        # 重新创建schema
        success, schema_name, password = importer.recreate_schema_from_sql(
            schema_name=database_name,
            sql_file_path=sql_dir_path + database_name.lower() + ".sql"
        )
        
        results[database_name] = (success, schema_name, password)
        
        if success:
            print(f"Schema重建成功: {schema_name}")
        else:
            print(f"Schema重建失败: {database_name}")
    
    return results

def recreate_database_original(database_name):
    """
    原始版本的recreate_database函数（每次都新建连接）
    保留以备需要
    """
    database_name = database_name.upper()
    
    # 每次创建新的连接实例
    importer = OracleSchemaImporter(
        username="system",
        password="MyPassword123", 
        host="localhost",
        port="1521",
        service_name="XEPDB1"
    )
    
    # 连接数据库
    if importer.connect():
        # 重新创建schema
        success, schema_name, password = importer.recreate_schema_from_sql(
            schema_name=database_name,
            sql_file_path=sql_dir_path + database_name.lower() + ".sql"
        )
        
        if success:
            print(f"Schema重建成功: {schema_name}")
        else:
            print("Schema重建失败!")
        
        # 断开连接
        importer.disconnect()
        
        return success, schema_name, password
    else:
        return False, None, None

def recreate_database_with_context(database_name, host="localhost", port="1521"):
    """
    使用上下文管理器的数据库重建函数（推荐使用）
    
    Args:
        database_name (str): 数据库名称
        host (str): Oracle数据库主机地址
        port (str): Oracle数据库端口号
    
    Returns:
        tuple: (success, schema_name, password)
    """
    database_name = database_name.upper()
    
    try:
        with OracleConnectionManager(host=host, port=port) as importer:
            success, schema_name, password = importer.recreate_schema_from_sql(
                schema_name=database_name,
                sql_file_path=sql_dir_path + database_name.lower() + ".sql"
            )
            
            return success, schema_name, password
    except ConnectionError as e:
        print(f"连接错误: {e}")
        return False, None, None
    except Exception as e:
        print(f"操作失败: {e}")
        return False, None, None

def recreate_databases_with_context(database_names, host="localhost", port="1521"):
    """
    使用上下文管理器的批量数据库重建函数（推荐使用）
    
    Args:
        database_names (list): 数据库名称列表
        host (str): Oracle数据库主机地址
        port (str): Oracle数据库端口号
    
    Returns:
        dict: {database_name: (success, schema_name, password)}
    """
    results = {}
    
    try:
        with OracleConnectionManager(host=host, port=port) as importer:
            for database_name in database_names:
                database_name = database_name.upper()
                
                success, schema_name, password = importer.recreate_schema_from_sql(
                    schema_name=database_name,
                    sql_file_path=sql_dir_path + database_name.lower() + ".sql"
                )
                
                results[database_name] = (success, schema_name, password)
        
        return results
    except ConnectionError as e:
        print(f"连接错误: {e}")
        return {name: (False, None, None) for name in database_names}
    except Exception as e:
        print(f"操作失败: {e}")
        return {name: (False, None, None) for name in database_names}

if __name__ == "__main__":
    # database_names = sorted(os.listdir(sql_dir_path))
    # database_names = [name.replace(".sql", "") for name in database_names]
    # recreate_databases_with_context(database_names)

    recreate_database_with_context("candidate_poll")
