import joblib
import chromadb
import json
import os
import sys

# Adicionar bloco_01 ao path para importar módulo 'classifier'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bloco_01'))
from classifier import IntentClassifier

class PipelineOrchestrator:
    def __init__(self, model_path, db_path, collection_name):
        """
        Inicializa o orquestrador carregando o modelo e conectando ao ChromaDB.
        """
        print("[Orquestrador] Inicializando pipeline...")
        
        # 1. Carregar o modelo de classificação (.joblib) usando o método correto
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado no caminho: {model_path}")
        self.model = IntentClassifier.load(model_path)
        print(f"[Orquestrador] Modelo carregado de: {model_path}")

        # 2. Conectar ao banco de dados vetorial (ChromaDB)
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"[Orquestrador] Conectado ao ChromaDB. Coleção: '{collection_name}'")
        except Exception as e:
            raise ConnectionError(f"Erro ao acessar a coleção '{collection_name}' no ChromaDB. Verifique se ela existe. Detalhes: {e}")

    def process(self, user_input, output_file="file.txt"):
        """
        Executa o pipeline completo para uma entrada do usuário.
        """
        print(f"\n[Orquestrador] Processando entrada: '{user_input}'")

        # PASSO 1: Classificação de Intenção
        # O método predict retorna um dicionário com label, rank, confidence, etc.
        result = self.model.predict(user_input)
        intent_label = result["label"]
        intent_rank = result["rank"]
        intent_confidence = result["confidence"]
        print(f"[Orquestrador] Intenção classificada como: {intent_label} (rank {intent_rank}, confiança {intent_confidence})")

        # PASSO 2: Busca Semântica no ChromaDB
        try:
            db_results = self.collection.query(
                query_texts=[user_input], 
                n_results=1
            )
            
            # Extrair os metadados (onde está o comando) e a distância (confiança da busca)
            metadata = db_results['metadatas'][0][0] if db_results['metadatas'] else {"erro": "Nenhum resultado encontrado"}
            distance = db_results['distances'][0][0] if db_results['distances'] else None
            
            # Adiciona a métrica de confiança aos metadados para o próximo modelo saber o quão certo é isso
            metadata["vector_search_distance"] = distance

        except Exception as e:
            metadata = {"erro": f"Falha na busca vetorial: {str(e)}"}
            print(f"[Orquestrador] Aviso: {metadata['erro']}")

        # PASSO 3: Combinar os dados
        # Criamos um dicionário estruturado que servirá como o "prompt" ou entrada para o seu próximo modelo
        combined_payload = {
            "entrada_original": user_input,
            "classificacao_intent": {
                "label": intent_label,
                "rank": intent_rank,
                "confidence": intent_confidence
            },
            "contexto_banco_dados": metadata
        }

        # PASSO 4: Salvar em arquivo para o próximo passo
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_payload, f, ensure_ascii=False, indent=4)
            
        print(f"[Orquestrador] Dados combinados salvos em: {output_file}")
        
        return combined_payload


# ==========================================
# EXECUÇÃO PRINCIPAL (Exemplo de Uso)
# ==========================================
if __name__ == "__main__":
    # Configuração dos caminhos (conforme você pediu)
    MODEL_PATH = "./modelos/model.joblib"
    DB_PATH = "./bloco_02/chroma_db"
    COLLECTION_NAME = "comandos_sistema" # O nome da coleção que criamos no passo anterior
    OUTPUT_FILE = "file.txt"

    try:
        # Inicia o orquestrador
        orchestrator = PipelineOrchestrator(MODEL_PATH, DB_PATH, COLLECTION_NAME)

        # Loop interativo para testes
        print("\n" + "="*50)
        print("Sistema pronto. Digite sua pergunta/comando (ou 'sair' para encerrar):")
        print("="*50)

        while True:
            user_input = input("\n>> ")
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando...")
                break
            
            if not user_input.strip():
                continue

            # Processa a entrada
            orchestrator.process(user_input, OUTPUT_FILE)
            
            # Opcional: Mostra na tela como o arquivo file.txt ficou
            print("\n--- Conteúdo gerado para o próximo modelo ---")
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                print(f.read())
            print("----------------------------------------------")

    except Exception as e:
        print(f"\n[ERRO FATAL] O orquestrador parou. Motivo: {e}")