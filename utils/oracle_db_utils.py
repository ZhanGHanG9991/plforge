import os
import subprocess
import oracledb
import pandas as pd
import sqlparse
from sqlparse import sql, tokens
from typing import Any, Dict, List, Optional
import re

from .oracle_utils import recreate_database_with_context

def get_tables_info(database_name, host="localhost", port=1521):
    """
    获取Oracle指定schema下的数据表信息
    
    Args:
        schema_name: Oracle schema名称
        host: Oracle数据库主机地址
        port: Oracle数据库端口号
        
    Returns:
        dict: 表名为key，列信息列表为value的字典
        格式: {'table_name': [('column_name', 'data_type'), ...]}
    """
    # Oracle连接信息
    username = "system"
    password = "MyPassword123"
    service_name = "XEPDB1"

    schema_name = database_name.upper()
    
    tables_info = {}
    
    try:
        # 建立连接 - oracledb支持直接传入参数
        with oracledb.connect(user=username, password=password, 
                             host=host, port=port, service_name=service_name) as conn:
            with conn.cursor() as cur:
                # 查询指定schema下的所有表名
                cur.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE owner = :schema_name
                    ORDER BY table_name
                """, schema_name=schema_name.upper())
                
                result = cur.fetchall()
                table_names = [table_name[0] for table_name in result]
                
                # 获取每个表的列信息
                for table_name in table_names:
                    cur.execute("""
                        SELECT column_name, data_type
                        FROM all_tab_columns 
                        WHERE owner = :schema_name AND table_name = :table_name
                        ORDER BY column_id
                    """, schema_name=schema_name.upper(), table_name=table_name)
                    
                    result = cur.fetchall()
                    # 直接使用Oracle原始数据类型，不做任何转换
                    tables_info[table_name] = [(col[0], col[1]) for col in result]
                    
    except oracledb.Error as e:
        print(f"Oracle数据库连接或查询出错: {e}")
        return {}
        
    return tables_info

def get_all_user_tables(schema_name, host="localhost", port=1521):
    schema_name = schema_name.upper()

    """获取Oracle数据库中指定schema内所有用户表的名称"""
    # Oracle连接信息
    username = "system"
    password = "MyPassword123"
    service_name = "XEPDB1"
    
    # 构建连接字符串
    dsn = f"{host}:{port}/{service_name}"
    
    with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
            # 获取指定schema中的所有用户表（排除系统表和回收站表）
            cur.execute("""
                SELECT table_name
                FROM all_tables
                WHERE owner = :schema_name
                AND table_name NOT LIKE 'BIN$%'
                ORDER BY table_name
            """, {'schema_name': schema_name})
            
            result = cur.fetchall()
            return [table_name[0] for table_name in result]

def get_important_system_tables():
    """返回需要监控的重要系统表列表"""
    return [
        'all_constraints',          # 约束信息  
        'all_triggers',             # 触发器信息
        'all_sequences',            # 序列信息
        'all_views'                 # 视图信息
    ]

def fetch_system_table_data(system_table, host="localhost", port=1521):
    """获取系统表数据，处理可能的权限或存在性问题"""
    try:
        # 解析连接信息
        username = "system"
        password = "MyPassword123"
        service_name = "XEPDB1"
        dsn = f"{host}:{port}/{service_name}"
        
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cursor() as cur:
                # 检查表/视图是否存在（检查all_tables和all_views）
                cur.execute("""
                    SELECT CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM all_tables 
                            WHERE table_name = :table_name
                        ) THEN 1
                        WHEN EXISTS (
                            SELECT 1 FROM all_views 
                            WHERE view_name = :table_name
                        ) THEN 1
                        ELSE 0
                    END as table_exists
                    FROM dual
                """, {'table_name': system_table.upper()})
                
                if not cur.fetchone()[0]:
                    return None
                
                # 尝试查询系统表数据
                # Oracle需要显式指定要排序的列，这里使用ROWNUM作为默认排序
                cur.execute(f"""
                    SELECT * FROM {system_table} 
                    WHERE ROWNUM <= 100
                    ORDER BY 1
                """)
                result = cur.fetchall()
                return result
                
    except Exception as e:
        print(f"Warning: Could not fetch data from system table {system_table}: {e}")
        return None
    
def check_plsql_executability(generated_plsql, call_plsqls, database_name, host="localhost", port=1521):
    database_name = database_name.upper()
    execution_error = None
    try:
        recreate_database_with_context(database_name, host, port)
        username = "system"
        password = "MyPassword123"
        service_name = "XEPDB1"
        dsn = f"{host}:{port}/{service_name}"
        
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cursor() as cur:
                # 使用system权限切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {database_name}")
                # 设置查询超时 - Oracle使用call_timeout参数
                cur.call_timeout = 2 * 1000  # timeout单位为毫秒
                
                cur.execute(generated_plsql)

                for call in call_plsqls:
                    cur.execute(call)
    except Exception as e:
        execution_error = str(e)
    
    return execution_error

def fetch_query_results(query):
    """执行查询并返回结果"""
    # Oracle连接信息
    username = "system"
    password = "MyPassword123"
    host = "localhost"
    port = 1521
    service_name = "XEPDB1"
    dsn = f"{host}:{port}/{service_name}"
    
    with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            return result
        
def compare_plsql(schema_name, plsql1, plsql2, call_plsqls, include_system_tables, host="localhost", port=1521):
    """
    比较两个PL/SQL代码的执行结果
    
    Args:
        schema_name: Oracle schema名称
        plsql1: 第一个PL/SQL代码
        plsql2: 第二个PL/SQL代码  
        call_plsqls: 调用PL/SQL的语句列表
        include_system_tables: 是否包含系统表比较
        host: Oracle数据库主机地址
        port: Oracle数据库端口号
    
    Returns:
        True or False
    """
    try:
        schema_name = schema_name.upper()
        
        # Oracle连接信息
        username = "system"
        password = "MyPassword123"
        service_name = "XEPDB1"
        dsn = f"{host}:{port}/{service_name}"
        
        # 获取所有用户表
        all_user_tables = get_all_user_tables(schema_name, host, port)
        print(f"Found {len(all_user_tables)} user tables to compare: {all_user_tables}")
        
        # 获取重要系统表列表
        important_system_tables = get_important_system_tables() if include_system_tables else []
        print(f"Will compare {len(important_system_tables)} system tables")
        
        # 第一次执行
        recreate_database_with_context(schema_name, host, port)
        
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql1)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()

        # 收集第一次执行后的数据
        user_tables_results1 = {}
        system_tables_results1 = {}
        
        # 获取所有用户表数据
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM "{table}" ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        user_tables_results1[table] = pd.DataFrame(result)
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        user_tables_results1[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results1[sys_table] = fetch_system_table_data(sys_table, host, port)
        
        # 第二次执行
        recreate_database_with_context(schema_name, host, port)
        
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cursor() as cur:
                # 切换到目标schema
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                cur.execute(plsql2)
                for call in call_plsqls:
                    cur.execute(call)
                conn.commit()

        # 收集第二次执行后的数据
        user_tables_results2 = {}
        system_tables_results2 = {}
        
        # 获取所有用户表数据
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema_name}")
                
                for table in all_user_tables:
                    try:
                        select_query = f'SELECT * FROM "{table}" ORDER BY 1'
                        cur.execute(select_query)
                        result = cur.fetchall()
                        user_tables_results2[table] = pd.DataFrame(result)
                    except Exception as e:
                        print(f"Warning: Could not fetch data from user table {table}: {e}")
                        user_tables_results2[table] = None
        
        # 获取系统表数据
        for sys_table in important_system_tables:
            system_tables_results2[sys_table] = fetch_system_table_data(sys_table, host, port)

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

def preprocess_plsql_for_ast(sql_text: str) -> str:
    """
    Preprocess PL/SQL text to normalize formatting and optional syntax
    before AST parsing
    """
    # Remove extra whitespace
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
    # Remove optional IN/OUT/INOUT keywords from parameter declarations
    # Match: (param_name IN datatype) -> (param_name datatype)
    sql_text = re.sub(r'\(\s*(\w+)\s+(IN|OUT|INOUT)\s+(\w+)\s*\)', 
                      r'(\1 \3)', sql_text, flags=re.IGNORECASE)
    
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
    
    # Ensure single space around keywords
    keywords = [
        'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'IS', 'AS', 'BEGIN', 'END',
        'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP', 'SELECT', 'FROM', 'WHERE',
        'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES', 'COUNT', 'NUMBER', 'VARCHAR2',
        'AND', 'OR', 'NOT', 'NULL'
    ]
    
    for keyword in keywords:
        # Add space before and after keywords
        pattern = r'\b' + re.escape(keyword) + r'\b'
        sql_text = re.sub(pattern, f' {keyword} ', sql_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces again
    sql_text = re.sub(r'\s+', ' ', sql_text.strip())
    
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

class HybridPLSQLASTParser:
    """PL/SQL AST Parser that works on preprocessed, normalized text"""
    
    def __init__(self):
        self.keywords = {
            'CREATE', 'OR', 'REPLACE', 'PROCEDURE', 'FUNCTION', 'IS', 'AS', 'BEGIN', 'END',
            'IF', 'THEN', 'ELSE', 'ELSIF', 'WHILE', 'FOR', 'LOOP', 'SELECT', 'FROM', 'WHERE',
            'INSERT', 'INTO', 'UPDATE', 'SET', 'DELETE', 'VALUES', 'COUNT', 'NUMBER', 'VARCHAR2',
            'CHAR', 'DATE', 'TIMESTAMP', 'DECLARE', 'CURSOR', 'EXCEPTION', 'WHEN', 'OTHERS',
            'RAISE', 'RETURN', 'EXIT', 'CONTINUE', 'CASE', 'AND', 'OR', 'NOT', 'NULL', 
            'TRUE', 'FALSE', 'COMMIT', 'ROLLBACK'
        }
    
    def normalize_token_value(self, token) -> str:
        """Normalize token value for semantic equivalence"""
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
            else:
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

def is_exact_match_hybrid(plsql1: str, plsql2: str, debug: bool = False) -> bool:
    """
    Hybrid approach: preprocess text for normalization, then use AST comparison
    
    Args:
        plsql1: First PL/SQL statement
        plsql2: Second PL/SQL statement  
        debug: Whether to show debug information
        
    Returns:
        True if semantically equivalent using AST comparison
    """
    if debug:
        print("=== Hybrid AST Comparison (Preprocess + AST) ===")
        print(f"Original SQL1: {plsql1}")
        print(f"Original SQL2: {plsql2}")
    
    # Step 1: Preprocess both SQL texts
    preprocessed1 = preprocess_plsql_for_ast(plsql1)
    preprocessed2 = preprocess_plsql_for_ast(plsql2)
    
    if debug:
        print(f"Preprocessed SQL1: {preprocessed1}")
        print(f"Preprocessed SQL2: {preprocessed2}")
    
    # Step 2: Parse into ASTs
    parser = HybridPLSQLASTParser()
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

# Convenience functions
def is_exact_match(plsql1: str, plsql2: str) -> bool:
    """Check semantic equivalence using hybrid approach (no debug output)"""
    return is_exact_match_hybrid(plsql1, plsql2, debug=False)

def debug_semantic_equivalence_ast(plsql1: str, plsql2: str) -> bool:
    """Check semantic equivalence using hybrid approach (with debug output)"""
    return is_exact_match_hybrid(plsql1, plsql2, debug=True)

if __name__ == "__main__":
    print(get_all_user_tables("candidate_poll"))

    print(len(fetch_system_table_data("all_views")))

    recreate_database_with_context("candidate_poll")

    print(check_plsql_executability("""CREATE OR REPLACE PROCEDURE sp(para_FacID NUMBER, para_actid NUMBER) IS BEGIN UPDATE \"Faculty_Participates_in\" SET \"FacID\" = para_FacID WHERE \"actid\" = para_actid; END;""",
                              ["BEGIN\n  sp(2192, 782);\n  commit;\nEND;",
                               "BEGIN\n  sp(7723, 790);\n  commit;\nEND;",
                               "BEGIN\n  sp(4230, 785);\n  commit;\nEND;"],
                               "activity_1"))
    
    print(compare_plsql("activity_1",
                        """CREATE OR REPLACE PROCEDURE sp(para_FacID NUMBER, para_actid NUMBER) IS BEGIN UPDATE \"Faculty_Participates_in\" SET \"FacID\" = para_FacID WHERE \"actid\" = para_actid; END;""",
                        """CREATE OR REPLACE PROCEDURE sp(para_FacID NUMBER, para_actid NUMBER) IS BEGIN UPDATE \"Faculty_Participates_in\" SET \"FacID\" = para_FacID WHERE \"actid\" = para_actid; END;""",
                        ["BEGIN\n  sp(2192, 782);\n  commit;\nEND;",
                         "BEGIN\n  sp(7723, 790);\n  commit;\nEND;",
                         "BEGIN\n  sp(4230, 785);\n  commit;\nEND;"],
                         True))
    
    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_count NUMBER; BEGIN SELECT COUNT(*) INTO record_count FROM "coach" WHERE "Gender" = para_Gender; IF record_count = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_coun NUMBER; BEGIN SELECT COUNT(*) INTO record_coun FROM "coach" WHERE "Gender" = para_Gender; IF record_coun = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END"""))

    print(is_exact_match(f"""CREATE OR REPLACE PROCEDURE sp(para_Gender VARCHAR2) IS record_count NUMBER; BEGIN SELECT COUNT(*) INTO record_count FROM "coach" WHERE "Gender" = para_Gender; IF record_count = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END""",
                         f"""CREATE OR REPLACE PROCEDURE sp(para_Gender IN VARCHAR2) IS record_coun NUMBER; BEGIN SELECT COUNT(*) INTO record_coun FROM "coach" WHERE "Gender" = para_Gender; IF record_coun = 0 THEN INSERT INTO "coach" ("Gender") VALUES (para_Gender); END IF; END"""))
