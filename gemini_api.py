import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image


# Localiza pasta
pasta_projeto = Path(__file__).resolve().parent

# Localiza .env
caminho_env = pasta_projeto / ".env"

# Carrega  .env
load_dotenv(dotenv_path=caminho_env)

# chave
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        f"A chave GEMINI_API_KEY não foi encontrada.\n"
        f"Arquivo procurado: {caminho_env}"
    )

client = genai.Client(api_key=api_key)


def classificar_imagem(caminho_imagem):

    imagem = Image.open(caminho_imagem)

    prompt = """
Analise a imagem fornecida e determine se ela é:

REAL: uma fotografia capturada do mundo real.

IA: uma imagem criada artificialmente por inteligência artificial.

Responda exclusivamente com uma das duas opções:

REAL
IA
"""

    max_tentativas = 4

    for tentativa in range(max_tentativas):

        try:

            resposta = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    prompt,
                    imagem
                ]
            )

            resultado = resposta.text.strip().upper()

            return resultado

        except Exception as erro:

            mensagem = str(erro)

            # Erro temporário do servidor
            if "503" in mensagem or "UNAVAILABLE" in mensagem:

                if tentativa < max_tentativas - 1:

                    espera = 5 * (tentativa + 1)

                    print(
                        f"Gemini temporariamente indisponível. "
                        f"Nova tentativa em {espera} segundos..."
                    )

                    time.sleep(espera)

                else:
                    raise

            else:
                # Se for outro tipo de erro, não tenta novamente
                raise