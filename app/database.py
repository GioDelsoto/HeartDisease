# DATABASE CREATIO
import mysql.connector
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_path = os.path.join(base_dir, '.env')

load_dotenv(dotenv_path=env_path)

def get_db_connection(database_name = None):
    """Estabelece a conexão com o banco de dados."""
    try:
        
        if database_name == None:
            print(os.environ.get('HOST_DB') , os.environ.get('USER_DB') , os.environ.get('PASSWORD_DB'))

            conn = mysql.connector.connect(
                host=os.environ.get('HOST_DB'),
                user=os.environ.get('USER_DB'),
                password=os.environ.get('PASSWORD_DB')
            )
        else:
            print(os.environ.get('HOST_DB') , os.environ.get('USER_DB') , os.environ.get('PASSWORD_DB'), database_name)

            conn = mysql.connector.connect(
                host=os.environ.get('HOST_DB'),
                user=os.environ.get('USER_DB'),
                password=os.environ.get('PASSWORD_DB'),
                database=database_name
            )
            
        return conn
    except mysql.connector.Error as err:
        print(f"Erro na conexão: {err}")
        return None

def close_db_connection(conn):
    """Fecha a conexão com o banco de dados."""
    if conn and conn.is_connected():
        conn.close()



## TABLE OPERATIONS
def create_table(data_file, table_name = 'table_name', dtype_list = [], database_name = None, bd_credentials = ''):
    
    data = pd.read_csv(data_file)
    
    conn = get_db_connection(database_name = database_name)
    mycursor = conn.cursor()
    
    columns = data.columns.values
       
    concatenate_var_names = [col +' ' +dtype for col, dtype in zip(columns, dtype_list)]
    
    #Join var name and dtypes to a single string
    string_columns = ','.join(concatenate_var_names)
    
    #Create the table
    mycursor.execute(f"CREATE TABLE {table_name} ({string_columns})")
    
    #Verify if creation was a success
    #verify_table_existence(table_name, mycursor)
    
    print('Table creation was successful')

    close_db_connection(conn)
     
    
def populate_table_from_csv(data_file, table_name = 'table_name', database_name = '', bd_credentials = ''):
    
    data = pd.read_csv(data_file)

    conn = get_db_connection(database_name = database_name)
    mycursor = conn.cursor()

    columns = data.columns.values

    #Populate dataset
    columns_string = "(" + ", ".join(columns) + ")"
    variable_string = "(" + ", ".join( ["%s"]*len(columns) ) + ")"
    
    def change_dtype(val):
        
        if isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            return float(val)
        else:
            return val
            
    for i in range( data.shape[0] ):
        #Transform row values into appropriate data type (numpy.int is not allowed)
        values = [ change_dtype( data[j].iloc[i] ) for col_num, j in enumerate(columns) ]
        
        #Add the data to database
        mycursor.execute(f"INSERT INTO {table_name} {columns_string} VALUES {variable_string}", values)
        conn.commit() #To commit the insertion
    
    close_db_connection(conn)

    

def verify_table_existence(table_name, mycursor):
    mycursor.execute(f"DESCRIBE {table_name}")
    check_table = []
    for x in mycursor:
       check_table.append(x) 
    if len(check_table)>0:
        print('Table creation was successful')
    else:
        print('Table creation failed')



def delete_table(table_name, database_name, bd_credentials):

    conn = get_db_connection(database_name = database_name)
    mycursor = conn.cursor()    

    delete_table_query = f"DROP TABLE IF EXISTS {table_name}"
    mycursor.execute(delete_table_query)
    print(f"Table {table_name} has been deleted (if it existed).")

    close_db_connection(conn)


def list_tables(database_name):
    
    conn = get_db_connection(database_name = database_name)
    mycursor = conn.cursor()  
    
    mycursor.execute("SHOW TABLES")
    tables = mycursor.fetchall()
    print("Tables:")
    tables_names = []
    for tab in tables:
        print(tab[0])
        tables_names.append(tab[0])
        
    close_db_connection(conn)

    
    return tables_names


def get_values_from_table(table_name, database_name, where_filter):
    
    conn = get_db_connection(database_name = database_name)
    mycursor = conn.cursor() 
    
    mycursor.execute(f"DESCRIBE {table_name}")
    columns = mycursor.fetchall()
    # Extract and print column names
    column_names = [column[0] for column in columns]
    
    if where_filter:
        mycursor.execute(f"SELECT * FROM {table_name} WHERE {where_filter}")
    else:
        mycursor.execute(f"SELECT * FROM {table_name}")
    data = []
    for x in mycursor:
        data.append(x)
    
    print("Data retrieved was successful")
    close_db_connection(conn)

    
    return data, column_names



## DATABASE OPERATIONS

def create_database(database_name, bd_credentials):
    
    conn = get_db_connection()
    mycursor = conn.cursor()
    
    create_db_query = f"CREATE DATABASE {database_name}"
    mycursor.execute(create_db_query)
    print(f"Database {database_name} has been created.")
    close_db_connection(conn)




def delete_database(database_name, bd_credentials):
    conn = get_db_connection()
    mycursor = conn.cursor()    

    delete_db_query = f"DROP DATABASE IF EXISTS {database_name}"
    mycursor.execute(delete_db_query)
    print(f"Database {database_name} has been deleted (if it existed).")
    close_db_connection(conn)


    
def list_database():
    
    conn = get_db_connection()
    mycursor = conn.cursor()  
    
    mycursor.execute("SHOW DATABASES")
    databases = mycursor.fetchall()
    print("Databases:")
    database_names = []
    for db_name in databases:
        print(db_name[0])
        database_names.append(db_name[0])
        
    close_db_connection(conn)

    
    return database_names

