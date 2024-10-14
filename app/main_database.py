import mysql.connector
from datetime import datetime

import pandas as pd
import numpy as np

from database_functions import *



bd_credentials = {'host':'localhost', 'user': 'root', 'passwd': 'admin'}

dtype_list = ['smallint']*14
dtype_list[9] = 'float'

db_list = list_database()
delete_database('testdatabase', bd_credentials = '')

#Create new database
database_name = 'database_heart_disease'
if not(database_name in db_list):
    create_database('database_heart_disease', bd_credentials)

#Create a new table
tables_list = list_tables(database_name)

table_name = 'heart_disease'

if not(table_name in tables_list):
    create_table('heart-disease.csv', table_name = 'heart_disease', dtype_list = dtype_list, database_name = 'database_heart_disease', bd_credentials = '')

if False:
    populate_table_from_csv('heart-disease.csv', table_name = 'heart_disease', database_name = 'database_heart_disease', bd_credentials = '')


#
data, columns = get_values_from_table(table_name = table_name, database_name = database_name, where_filter = 'target = 1')