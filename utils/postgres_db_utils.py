import os
import psycopg
import subprocess
import pandas as pd
import sqlparse
from sqlparse import sql, tokens
from typing import Any, Dict, List, Optional
import re

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

def get_all_user_tables(database_name):
    """获取数据库中所有用户表的名称"""
    conn_db_info = f"""host=localhost dbname={database_name} user=postgres password=123456"""
    
    with psycopg.connect(conn_db_info) as conn:
        with conn.cursor() as cur:
            # 获取所有用户表（排除系统表）
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tablename;
            """)
            result = cur.fetchall()
            return [table_name[0] for table_name in result]

def get_important_system_tables():
    """返回需要监控的重要系统表列表"""
    return [
        'pg_indexes',               # 索引信息
        'pg_constraints',           # 约束信息
        'pg_triggers',              # 触发器信息
        'pg_sequences',             # 序列信息
        'pg_views',                 # 视图信息
        'pg_user_mappings',         # 用户映射
        'pg_policies',              # 行级安全策略
        'pg_rules'                  # 规则信息
    ]

def fetch_system_table_data(database_conn_info, system_table):
    """获取系统表数据，处理可能的权限或存在性问题"""
    try:
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 检查表是否存在
                cur.execute(f"""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM information_schema.tables 
                        WHERE table_name = '{system_table}'
                        AND table_schema IN ('pg_catalog', 'information_schema')
                    ) OR EXISTS (
                        SELECT 1 
                        FROM information_schema.views 
                        WHERE table_name = '{system_table}'
                        AND table_schema IN ('pg_catalog', 'information_schema')
                    );
                """)
                
                if not cur.fetchone()[0]:
                    return None
                
                # 尝试查询系统表数据
                cur.execute(f"SELECT * FROM {system_table} ORDER BY 1;")
                result = cur.fetchall()
                return result
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None

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
            # 设置查询超时
            cur.execute(f"SET statement_timeout = {2 * 1000};")  # timeout单位为毫秒
            cur.execute(sql)

def check_plsql_executability(generated_plsql, call_plsqls, database_name):
    execution_error = None
    try:
        database_conn_info = f"""host=localhost user=postgres password=123456 dbname={database_name}"""
        restore_databases(conn_info, host, port, user, password, [database_name])
        with psycopg.connect(database_conn_info) as conn:
            with conn.cursor() as cur:
                # 设置查询超时
                cur.execute(f"SET statement_timeout = {2 * 1000};")  # timeout单位为毫秒
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

def compare_plsql(database_name, plsql1, plsql2, call_plsqls, include_system_tables):
    """
    比较两个PL/SQL代码的执行结果
    
    Args:
        database_name: 数据库名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表比较
    
    Returns:
        True or False
    """
    try:
        database_conn_info = f"""host=localhost user=postgres password=123456 dbname={database_name}"""
        
        # 获取所有用户表
        all_user_tables = get_all_user_tables(database_name)
        print(f"Found {len(all_user_tables)} user tables to compare: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Will compare {len(important_system_tables)} system tables")
        
        # 第一次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql1)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        # 收集第一次执行后的数据
        user_tables_results1 = {}
        system_tables_results1 = {}
        
        # 获取所有用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results1[table] = pd.DataFrame(result)
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table}: {e}")
                user_tables_results1[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(database_conn_info, sys_table)
        
        # 第二次执行
        restore_databases(conn_info, host, port, user, password, [database_name])
        execute_sql(database_conn_info, plsql2)
        for call in call_plsqls:
            execute_sql(database_conn_info, call)

        # 收集第二次执行后的数据
        user_tables_results2 = {}
        system_tables_results2 = {}
        
        # 获取所有用户表数据
        for table in all_user_tables:
            try:
                select_query = f'SELECT * FROM "{table}" ORDER BY 1;'
                result = fetch_query_results(database_conn_info, select_query)
                user_tables_results2[table] = pd.DataFrame(result)
            except Exception as e:
                print(f"Warning: Could not fetch data from user table {table}: {e}")
                user_tables_results2[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(database_conn_info, sys_table)

        # 比较用户表数据
        user_tables_same = True
        user_tables_diff = []
        
        for table in all_user_tables:
            df1 = user_tables_results1.get(table)
            df2 = user_tables_results2.get(table)
            
            if df1 is None and df2 is None:
                continue
            elif df1 is None or df2 is None:
                user_tables_same = False
                user_tables_diff.append(table)
            elif not df1.equals(df2):
                user_tables_same = False
                user_tables_diff.append(table)
        
        # 比较系统表数据
        system_tables_same = True
        system_tables_diff = []
        
        if include_system_tables:
            for sys_table in important_system_tables:
                result1 = system_tables_results1.get(sys_table)
                result2 = system_tables_results2.get(sys_table)
                
                if result1 is None and result2 is None:
                    continue
                elif result1 is None or result2 is None:
                    system_tables_same = False
                    system_tables_diff.append(sys_table)
                elif result1 != result2:
                    system_tables_same = False
                    system_tables_diff.append(sys_table)
        
        # 综合结果
        overall_same = user_tables_same and system_tables_same
        
        result = {
            'overall_same': overall_same,
            'user_tables_same': user_tables_same,
            'system_tables_same': system_tables_same,
            'user_tables_compared': len(all_user_tables),
            'system_tables_compared': len(important_system_tables),
            'user_tables_diff': user_tables_diff,
            'system_tables_diff': system_tables_diff
        }

        print(result)
        
        return result.get('overall_same', False)
    
    except Exception as e:
        print(f"Error in compare_plsql: {e}")
        return False
    
"""
PostgreSQL PL/pgSQL Semantic Equivalence Checker

This module provides tools for comparing two PL/pgSQL code blocks to determine if they are
semantically equivalent, even when they differ in:
- Whitespace and formatting
- Variable names (identifiers)
- Parameter names
- Cursor names  
- Code structure spacing

The tool uses a hybrid approach:
1. Text preprocessing to normalize syntax and formatting
2. Abstract Syntax Tree (AST) parsing using sqlparse
3. Semantic comparison of normalized AST structures

Key Features:
- Handles PostgreSQL-specific syntax ($$ delimiters, LANGUAGE plpgsql, etc.)
- Abstracts away user-defined identifiers while preserving system objects
- Normalizes PostgreSQL data types and keywords
- Supports complex PL/pgSQL constructs (cursors, exceptions, loops, etc.)

Usage:
    from postgres_db_utils import is_exact_match
    
    code1 = '''CREATE OR REPLACE PROCEDURE sp(param1 text) LANGUAGE plpgsql AS $$
               DECLARE cursor1 CURSOR FOR SELECT * FROM table1;
               BEGIN ... END; $$;'''
    
    code2 = '''CREATE OR REPLACE PROCEDURE sp(param2 text) LANGUAGE plpgsql AS $$
               DECLARE cursor2 CURSOR FOR SELECT * FROM table1;  
               BEGIN ... END; $$;'''
    
    result = is_exact_match(code1, code2)  # Returns True - semantically equivalent
"""

def preprocess_plpgsql_for_ast(sql_text: str) -> str:
    """
    Preprocess PL/pgSQL text to normalize formatting and optional syntax
    before AST parsing
    """
    # Remove extra whitespace
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
    # Handle PostgreSQL's $$ delimiter by temporarily replacing it
    # This helps normalize the function body
    dollar_pattern = r'\$\$([^$]*)\$\$'
    sql_text = re.sub(dollar_pattern, r'<BODY>\1</BODY>', sql_text)
    
    # Normalize LANGUAGE clause - make it consistent
    sql_text = re.sub(r'\s+LANGUAGE\s+plpgsql\s+', ' LANGUAGE plpgsql ', sql_text, flags=re.IGNORECASE)
    
    # Remove optional parameter modes (IN/OUT/INOUT) from parameter declarations
    # Match: (param_name IN datatype) -> (param_name datatype)
    sql_text = re.sub(r'\(\s*(\w+)\s+(IN|OUT|INOUT)\s+(\w+)\s*\)', 
                      r'(\1 \3)', sql_text, flags=re.IGNORECASE)
    
    # Normalize PostgreSQL data types
    # TEXT, VARCHAR, INTEGER, etc.
    pg_types = ['TEXT', 'VARCHAR', 'INTEGER', 'INT', 'BIGINT', 'SMALLINT', 
                'DECIMAL', 'NUMERIC', 'REAL', 'DOUBLE PRECISION', 'BOOLEAN', 
                'DATE', 'TIME', 'TIMESTAMP', 'TIMESTAMPTZ', 'INTERVAL', 'UUID']
    
    for pg_type in pg_types:
        # Normalize type declarations
        pattern = r'\b' + re.escape(pg_type.lower()) + r'\b'
        sql_text = re.sub(pattern, pg_type, sql_text, flags=re.IGNORECASE)
    
    # Normalize variable declarations: var_name datatype; -> var_name datatype;
    sql_text = re.sub(r'(\w+)\s+(%TYPE|%ROWTYPE)', r'\1\2', sql_text, flags=re.IGNORECASE)
    
    # Normalize exception handling
    sql_text = re.sub(r'\s+EXCEPTION\s+WHEN\s+', ' EXCEPTION WHEN ', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+WHEN\s+OTHERS\s+', ' WHEN OTHERS ', sql_text, flags=re.IGNORECASE)
    
    # Normalize cursor operations
    sql_text = re.sub(r'\s+CURRENT\s+OF\s+', ' CURRENT OF ', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+EXIT\s+WHEN\s+', ' EXIT WHEN ', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+NOT\s+FOUND\s*', ' NOT FOUND ', sql_text, flags=re.IGNORECASE)
    
    # Normalize function calls - remove spaces in function calls
    # COUNT ( * ) -> COUNT(*)
    sql_text = re.sub(r'(\w+)\s*\(\s*\*\s*\)', r'\1(*)', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'(\w+)\s*\(\s*([^)]+)\s*\)', r'\1(\2)', sql_text)
    
    # Normalize parentheses spacing
    sql_text = re.sub(r'\s*\(\s*', '(', sql_text)
    sql_text = re.sub(r'\s*\)\s*', ')', sql_text)
    
    # Normalize punctuation spacing
    sql_text = re.sub(r'\s*;\s*', ';', sql_text)
    sql_text = re.sub(r'\s*,\s*', ',', sql_text)
    
    # Normalize operators spacing
    sql_text = re.sub(r'\s*=\s*', '=', sql_text)
    sql_text = re.sub(r'\s*>\s*', '>', sql_text)
    sql_text = re.sub(r'\s*<\s*', '<', sql_text)
    sql_text = re.sub(r'\s*:=\s*', ':=', sql_text)  # PL/pgSQL assignment
    
    # Normalize string literals - preserve quoted identifiers
    sql_text = re.sub(r'\s*"\s*([^"]+)\s*"\s*', r'"\1"', sql_text)
    
    # Ensure single space around keywords
    keywords = [
        'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'LANGUAGE', 'plpgsql', 'AS',
        'DECLARE', 'BEGIN', 'END', 'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP',
        'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES',
        'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE', 'CURSOR', 'RECORD', 'OPEN', 'CLOSE',
        'FETCH', 'EXIT', 'WHEN', 'FOUND', 'CURRENT', 'OF', 'EXCEPTION', 'RAISE', 'RETURN',
        'COMMIT', 'ROLLBACK', 'RETURNS', 'RETURN'
    ]
    
    for keyword in keywords:
        # Add space before and after keywords
        pattern = r'\b' + re.escape(keyword) + r'\b'
        sql_text = re.sub(pattern, f' {keyword} ', sql_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces again
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
    # Restore $$ delimiters but normalize them
    sql_text = re.sub(r'<BODY>([^<]*)</BODY>', r'$$\1$$', sql_text)
    
    return sql_text

class ASTNode:
    """Abstract Syntax Tree Node"""
    def __init__(self, node_type: str, value: Optional[str] = None, children: Optional[List['ASTNode']] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
    
    def __repr__(self):
        if self.value and not self.children:
            return f"{self.node_type}({self.value})"
        elif self.children:
            children_str = ', '.join(str(child) for child in self.children)
            if self.value:
                return f"{self.node_type}({self.value})[{children_str}]"
            else:
                return f"{self.node_type}[{children_str}]"
        else:
            return self.node_type
    
    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (self.node_type == other.node_type and 
                self.value == other.value and 
                self.children == other.children)

class HybridPLpgSQLASTParser:
    """PL/pgSQL AST Parser that works on preprocessed, normalized text"""
    
    def __init__(self):
        self.keywords = {
            'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'LANGUAGE', 'plpgsql', 'AS', 
            'DECLARE', 'BEGIN', 'END', 'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP',
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES',
            'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE', 'CURSOR', 'RECORD', 'OPEN', 'CLOSE',
            'FETCH', 'EXIT', 'WHEN', 'FOUND', 'CURRENT', 'OF', 'EXCEPTION', 'RAISE', 'RETURN',
            'COMMIT', 'ROLLBACK', 'RETURNS', 'RETURN', 'VOLATILE', 'STABLE', 'IMMUTABLE',
            # PostgreSQL data types
            'TEXT', 'VARCHAR', 'INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'DECIMAL', 'NUMERIC',
            'REAL', 'DOUBLE', 'PRECISION', 'BOOLEAN', 'DATE', 'TIME', 'TIMESTAMP', 'TIMESTAMPTZ',
            'INTERVAL', 'UUID', 'JSON', 'JSONB', 'BYTEA', 'SERIAL', 'BIGSERIAL',
            # PostgreSQL specific keywords
            'PERFORM', 'GET', 'DIAGNOSTICS', 'ROWTYPE', 'TYPE', 'ARRAY', 'SLICE',
            'NOTICE', 'WARNING', 'STRICT', 'CONTINUE', 'CASE', 'USING', 'EXECUTE',
            'FOREACH', 'REVERSE', 'BY', 'CONCURRENTLY', 'CONFLICT', 'NOTHING', 'EXCLUDED'
        }
        # For variable name abstraction - track different types of identifiers
        self.table_names = set()
        self.column_names = set()
        self.variable_names = set()
        self.cursor_names = set()
        self.parameter_names = set()
    
    def is_system_object(self, name: str) -> bool:
        """Check if identifier refers to system objects that shouldn't be abstracted"""
        system_objects = {
            'FOUND', 'RECORD', 'SQLSTATE', 'SQLERRM', 'ROW_COUNT', 
            'CURRENT_USER', 'SESSION_USER', 'CURRENT_TIMESTAMP', 'NOW'
        }
        return name.upper() in system_objects
    
    def normalize_token_value(self, token) -> str:
        """Normalize token value for semantic equivalence with variable abstraction"""
        if not hasattr(token, 'value') or not token.value:
            return str(token)
        
        value = token.value.strip()
        
        # Normalize different token types
        if token.ttype in tokens.Literal.Number:
            return "<NUMBER>"
        elif token.ttype in tokens.Literal.String:
            return "<STRING>"
        elif (token.ttype in tokens.Name or 
              token.ttype in tokens.Name.Builtin or
              isinstance(token, sql.Identifier)):
            upper_value = value.upper()
            if upper_value in self.keywords:
                return upper_value
            elif self.is_system_object(value):
                return upper_value
            else:
                # Abstract away user-defined identifiers
                return "<IDENTIFIER>"
        elif token.ttype in tokens.Keyword:
            return value.upper()
        elif token.ttype in tokens.Punctuation:
            return value
        elif token.ttype in tokens.Operator:
            return value
        else:
            return value.upper()
    
    def parse_token_to_ast(self, token) -> Optional[ASTNode]:
        """Convert sqlparse token to AST node"""
        if token.is_whitespace:
            return None
            
        if isinstance(token, sql.Statement):
            return self.parse_statement(token)
        elif isinstance(token, sql.Parenthesis):
            return self.parse_parenthesis(token)
        elif isinstance(token, sql.Function):
            return self.parse_function(token)
        elif isinstance(token, sql.Identifier):
            return self.parse_identifier(token)
        elif isinstance(token, sql.IdentifierList):
            return self.parse_identifier_list(token)
        elif isinstance(token, sql.Comparison):
            return self.parse_comparison(token)
        elif isinstance(token, sql.Where):
            return self.parse_where(token)
        elif hasattr(token, 'tokens') and token.tokens:
            return self.parse_group(token)
        else:
            return self.parse_terminal(token)
    
    def parse_statement(self, stmt) -> ASTNode:
        """Parse SQL statement"""
        children = []
        for token in stmt.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("STATEMENT", children=children)
    
    def parse_parenthesis(self, paren) -> ASTNode:
        """Parse parentheses group"""
        children = []
        for token in paren.tokens:
            if token.value not in ['(', ')']:
                child = self.parse_token_to_ast(token)
                if child:
                    children.append(child)
        return ASTNode("PARENTHESIS", children=children)
    
    def parse_function(self, func) -> ASTNode:
        """Parse function call"""
        children = []
        for token in func.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("FUNCTION", children=children)
    
    def parse_identifier(self, ident) -> ASTNode:
        """Parse identifier"""
        normalized = self.normalize_token_value(ident)
        return ASTNode("IDENTIFIER", value=normalized)
    
    def parse_identifier_list(self, ident_list) -> ASTNode:
        """Parse identifier list"""
        children = []
        for token in ident_list.tokens:
            if token.value != ',':
                child = self.parse_token_to_ast(token)
                if child:
                    children.append(child)
        return ASTNode("IDENTIFIER_LIST", children=children)
    
    def parse_comparison(self, comp) -> ASTNode:
        """Parse comparison expression"""
        children = []
        for token in comp.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("COMPARISON", children=children)
    
    def parse_where(self, where) -> ASTNode:
        """Parse WHERE clause"""
        children = []
        for token in where.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        return ASTNode("WHERE", children=children)
    
    def parse_group(self, group) -> ASTNode:
        """Parse generic token group"""
        children = []
        for token in group.tokens:
            child = self.parse_token_to_ast(token)
            if child:
                children.append(child)
        
        # Determine group type based on first significant child
        if children:
            first_child = children[0]
            if (first_child.node_type == "TERMINAL" and 
                first_child.value and 
                isinstance(first_child.value, str)):
                value = first_child.value.upper()
                if value == "SELECT":
                    return ASTNode("SELECT_STATEMENT", children=children)
                elif value == "INSERT":
                    return ASTNode("INSERT_STATEMENT", children=children)
                elif value == "IF":
                    return ASTNode("IF_STATEMENT", children=children)
        
        return ASTNode("GROUP", children=children)
    
    def parse_terminal(self, token) -> ASTNode:
        """Parse terminal token"""
        normalized = self.normalize_token_value(token)
        return ASTNode("TERMINAL", value=normalized)
    
    def parse_sql(self, sql_text: str) -> List[ASTNode]:
        """Parse preprocessed SQL text into AST"""
        parsed = sqlparse.parse(sql_text)
        asts = []
        for stmt in parsed:
            ast = self.parse_token_to_ast(stmt)
            if ast:
                asts.append(ast)
        return asts

def compare_ast_nodes(node1: ASTNode, node2: ASTNode) -> bool:
    """Recursively compare two AST nodes"""
    # Compare node types
    if node1.node_type != node2.node_type:
        return False
    
    # Compare values
    if node1.value != node2.value:
        return False
    
    # Compare children count
    if len(node1.children) != len(node2.children):
        return False
    
    # Recursively compare children
    for child1, child2 in zip(node1.children, node2.children):
        if not compare_ast_nodes(child1, child2):
            return False
    
    return True

def is_exact_match_hybrid_plpgsql(plpgsql1: str, plpgsql2: str, debug: bool = False) -> bool:
    """
    Hybrid approach for PL/pgSQL: preprocess text for normalization, then use AST comparison
    
    Args:
        plpgsql1: First PL/pgSQL statement
        plpgsql2: Second PL/pgSQL statement  
        debug: Whether to show debug information
        
    Returns:
        True if semantically equivalent using AST comparison
    """
    if debug:
        print("=== Hybrid AST Comparison for PL/pgSQL (Preprocess + AST) ===")
        print(f"Original SQL1: {plpgsql1}")
        print(f"Original SQL2: {plpgsql2}")
    
    # Step 1: Preprocess both SQL texts
    preprocessed1 = preprocess_plpgsql_for_ast(plpgsql1)
    preprocessed2 = preprocess_plpgsql_for_ast(plpgsql2)
    
    if debug:
        print(f"Preprocessed SQL1: {preprocessed1}")
        print(f"Preprocessed SQL2: {preprocessed2}")
    
    # Step 2: Parse into ASTs
    parser = HybridPLpgSQLASTParser()
    ast1 = parser.parse_sql(preprocessed1)
    ast2 = parser.parse_sql(preprocessed2)
    
    if debug:
        print(f"AST1 count: {len(ast1)}")
        print(f"AST2 count: {len(ast2)}")
    
    # Step 3: Compare AST count
    if len(ast1) != len(ast2):
        if debug:
            print("Different number of AST trees")
        return False
    
    # Step 4: Compare each AST tree
    for i, (tree1, tree2) in enumerate(zip(ast1, ast2)):
        if debug:
            print(f"\n--- AST Tree {i+1} ---")
            print(f"Tree1: {tree1}")
            print(f"Tree2: {tree2}")
        
        if not compare_ast_nodes(tree1, tree2):
            if debug:
                print(f"AST trees {i+1} do not match")
            return False
        elif debug:
            print(f"AST trees {i+1} match!")
    
    return True

# Convenience functions for PL/pgSQL
def is_exact_match(plpgsql1: str, plpgsql2: str) -> bool:
    """Check semantic equivalence of PL/pgSQL using hybrid approach (no debug output)"""
    try:
        return is_exact_match_hybrid_plpgsql(plpgsql1, plpgsql2, debug=False)
    except Exception:
        return False

def debug_semantic_equivalence_ast(plpgsql1: str, plpgsql2: str) -> bool:
    """Check semantic equivalence of PL/pgSQL using hybrid approach (with debug output)"""
    return is_exact_match_hybrid_plpgsql(plpgsql1, plpgsql2, debug=True)

if __name__ == "__main__":
    print("=== PL/pgSQL Semantic Equivalence Tests ===\n")
    
    # Test 1: Different cursor names - should return True
    print("Test 1: Different cursor names (should be True)")
    result1 = is_exact_match(
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
           CLOSE ref_cursor; END; $$;""",
        
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cur; END LOOP; 
           CLOSE ref_cur; END; $$;"""
    )
    print(f"Result: {result1}\n")
    
    # Test 2: Different formatting and spacing - should return True
    print("Test 2: Different formatting and spacing (should be True)")
    result2 = is_exact_match(
        """CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql AS $$ 
           DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD; 
           BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cursor; END LOOP; 
           CLOSE ref_cursor; END; $$;""",
        
        """CREATE OR REPLACE PROCEDURE sp (para_state text, para_city text, para_bname text) 
           LANGUAGE plpgsql   AS $$  
           DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE "state" > para_state AND "bname" < para_bname; 
           rec RECORD;   
           BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; 
           UPDATE "bank" SET "city" = para_city WHERE CURRENT OF ref_cur; END LOOP; 
           CLOSE ref_cur; END; $$;"""
    )
    print(f"Result: {result2}\n")

    # print("expect True")
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cursor; END LOOP; CLOSE ref_cursor; END; $$;""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cur; END LOOP; CLOSE ref_cur; END; $$;"""))
    
    # print("expect True")
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_state text, para_city text, para_bname text) LANGUAGE plpgsql AS $$ DECLARE ref_cursor CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD; BEGIN OPEN ref_cursor; LOOP FETCH ref_cursor INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cursor; END LOOP; CLOSE ref_cursor; END; $$;""",
                         f"""CREATE OR REPLACE PROCEDURE sp (para_state text, para_city text, para_bname text) LANGUAGE plpgsql   AS $$  \nDECLARE ref_cur CURSOR FOR SELECT * FROM bank WHERE \"state\" > para_state AND \"bname\" < para_bname; rec RECORD;   BEGIN OPEN ref_cur; LOOP FETCH ref_cur INTO rec; EXIT WHEN NOT FOUND; UPDATE \"bank\" SET \"city\" = para_city WHERE CURRENT OF ref_cur; END LOOP; CLOSE ref_cur; END; $$;"""))