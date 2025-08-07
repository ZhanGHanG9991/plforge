import os
import psycopg
import subprocess
import pandas as pd
import sqlparse
    
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

# def get_db_schema_sequence(schema):
#     schema_sequence = "database schema :\n"
#     for table in schema["schema_items"]:
#         table_name, table_comment = table["table_name"], table["table_comment"]
#         table_info = "table " + table_name + " , columns = [ "
#         for column_name in table["column_names"]:
#             table_info += (column_name + " , ")
#         table_info = table_info[:-2] + " ]\n"
#         schema_sequence += table_info
#     return schema_sequence.strip()
