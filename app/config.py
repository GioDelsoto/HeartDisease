# app/config.py
import os
from dotenv import load_dotenv
# Carrega as variáveis do arquivo .env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_path = os.path.join(base_dir, '.env')

load_dotenv(dotenv_path=env_path)

# Agora você pode acessar as variáveis de ambiente
class Config:
    
    HOST_DB=os.environ.get('HOST_DB')
    USER_DB=os.environ.get('USER_DB')
    PASSWORD_DF=os.environ.get('PASSWORD_DF')

    SAVE_MODEL=True
    OVERWRITE_MODEL=False