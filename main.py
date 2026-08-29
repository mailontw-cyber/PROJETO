import time
from processamento import aplicar_processamentos
from gemini_api import classificar_imagem


print("======================================")
print(" TCC - CLASSIFICAÇÃO DE IMAGENS IA")
print("======================================")

caminho = "imagens/cachorro2.png"


try:
    
    #  IMAGEM ORIGINAL


    print("\nClassificando imagem ORIGINAL...")
    resultado_original = classificar_imagem(caminho)
    print(f"Original: {resultado_original}")


    
    # APLICAR PROCESSAMENTOS
    

    print("\nAplicando processamentos...")
    imagens_processadas = aplicar_processamentos(caminho)
    print("Processamentos concluídos!")


    # CLASSIFICA CADA IMAGEM PROCESSADA
   

    resultados = {}

    for tecnica, arquivo in imagens_processadas.items():
        print(f"\nClassificando: {tecnica}...")
        classificacao = classificar_imagem(arquivo)
        resultados[tecnica] = classificacao
        print(f"{tecnica}: {classificacao}")

    time.sleep(2)

    # RESULTADO

    print("\n======================================")
    print(" RESULTADOS")
    print("======================================")
  
    print(f"Original   : {resultado_original}")

    for tecnica, classificacao in resultados.items():
        print(f"{tecnica:<10} : {classificacao}")


except Exception as erro:

    print("\nOcorreu um erro:")
    print(erro)