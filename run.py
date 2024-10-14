from app import create_app
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))


# Cria uma instância da aplicação Flask
app = create_app()

if __name__ == '__main__':
    # Executa o servidor Flask
    app.run(debug=False, host='0.0.0.0')