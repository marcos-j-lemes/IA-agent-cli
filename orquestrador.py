import joblib
import chromadb
import json
import os
import sys
import subprocess
from datetime import datetime

ROOT_DIR = os.path.dirname(__file__)

# Adicionar bloco_01 ao path para importar módulo 'classifier'
sys.path.insert(0, os.path.join(ROOT_DIR, 'bloco_01'))
from classifier import IntentClassifier

# Adicionar bloco_03 ao path para importar o agente
sys.path.insert(0, os.path.join(ROOT_DIR, 'bloco_03'))
from agente import Agent

class PipelineOrchestrator:
    def __init__(self, model_path, db_path, collection_name, bloco03_dir=None, memory_file=None):
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

        # 3. Conectar ao Bloco 03 agente
        self.agent = None
        if bloco03_dir:
            if not os.path.isdir(bloco03_dir):
                raise FileNotFoundError(f"Diretório do bloco_03 não encontrado: {bloco03_dir}")
            sys.path.insert(0, bloco03_dir)
            self.agent = Agent(classifier_path=model_path, verbose=False)
            print(f"[Orquestrador] Agente do bloco_03 carregado de: {bloco03_dir}")

        # 4. Memória de comandos executados
        self.memory_path = memory_file or os.path.join(ROOT_DIR, "command_memory.json")
        self.history = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_memory(self):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def _last_execution_for_command(self, command: str):
        for entry in reversed(self.history):
            if entry.get("command") == command and entry.get("executed"):
                return entry
        return None

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

        # PASSO 5: Enviar também para o agente do bloco_03
        agent_result = None
        if self.agent is not None:
            agent_result = self.agent.process(user_input)
            agent_output_file = os.path.join(ROOT_DIR, 'bloco_03', 'agent_output.json')
            with open(agent_output_file, "w", encoding="utf-8") as f:
                json.dump(agent_result, f, ensure_ascii=False, indent=4)
            print(f"[Orquestrador] Saída do bloco_03 salva em: {agent_output_file}")

        # PASSO 6: Verificar histórico e executar se apropriado
        execution_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input": user_input,
            "command": None,
            "decision": None,
            "executed": False,
            "output": None,
            "error": None,
            "already_executed": False,
        }

        execution_result = None
        if agent_result is not None:
            command = agent_result.get("command")
            decision = agent_result.get("decision")
            execution_record["command"] = command
            execution_record["decision"] = decision

            previous = self._last_execution_for_command(command)
            if previous is not None:
                execution_record["already_executed"] = True
                execution_record["output"] = previous.get("output")
                execution_record["error"] = previous.get("error")
                print(f"[Orquestrador] Comando já executado anteriormente: {command}")
            elif decision == "EXECUTE":
                print(f"[Orquestrador] Executando comando: {command}")
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                execution_record["executed"] = True
                execution_record["output"] = result.stdout.strip()
                execution_record["error"] = result.stderr.strip()
                execution_result = {
                    "stdout": execution_record["output"],
                    "stderr": execution_record["error"],
                    "returncode": result.returncode,
                }
                if result.returncode == 0:
                    print(f"[Orquestrador] Comando executado com sucesso.")
                else:
                    print(f"[Orquestrador] Comando retornou código {result.returncode}.")
            elif decision == "CONFIRM":
                answer = input(f"[Orquestrador] Comando sugerido: {command}. Deseja executar? (s/n): ").strip().lower()
                if answer in ("s", "sim", "y", "yes"):
                    print(f"[Orquestrador] Executando comando confirmado pelo usuário.")
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    execution_record["executed"] = True
                    execution_record["output"] = result.stdout.strip()
                    execution_record["error"] = result.stderr.strip()
                    execution_result = {
                        "stdout": execution_record["output"],
                        "stderr": execution_record["error"],
                        "returncode": result.returncode,
                    }
                    if result.returncode == 0:
                        print(f"[Orquestrador] Comando executado com sucesso.")
                    else:
                        print(f"[Orquestrador] Comando retornou código {result.returncode}.")
                else:
                    print("[Orquestrador] Execução cancelada pelo usuário.")
            else:
                print("[Orquestrador] Decisão do agente foi REJECT: nenhum comando será executado.")

        self.history.append(execution_record)
        self._save_memory()
        print(f"[Orquestrador] Histórico salvo em: {self.memory_path}")

        return {
            "pipeline_payload": combined_payload,
            "bloco_03_output": agent_result,
            "execution": execution_record,
            "execution_result": execution_result,
        }


# ==========================================
# EXECUÇÃO PRINCIPAL (Exemplo de Uso)
# ==========================================
if __name__ == "__main__":
    # Configuração dos caminhos (conforme você pediu)
    MODEL_PATH = "./modelos/model.joblib"
    DB_PATH = "./bloco_02/chroma_db"
    COLLECTION_NAME = "comandos_sistema" # O nome da coleção que criamos no passo anterior
    BLOCO03_DIR = "./bloco_03"
    OUTPUT_FILE = "file.txt"

    try:
        # Inicia o orquestrador
        orchestrator = PipelineOrchestrator(MODEL_PATH, DB_PATH, COLLECTION_NAME, bloco03_dir=BLOCO03_DIR)

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
            result = orchestrator.process(user_input, OUTPUT_FILE)
            
            print("\n--- Saída do orquestrador (file.txt) ---")
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                print(f.read())
            print("----------------------------------------------")

            if result.get("bloco_03_output") is not None:
                print("\n--- Saída do bloco_03 ---")
                print(json.dumps(result["bloco_03_output"], ensure_ascii=False, indent=4))
                print("----------------------------------------------")

            if result.get("execution") is not None:
                print("\n--- Histórico / Execução ---")
                print(json.dumps(result["execution"], ensure_ascii=False, indent=4))
                if result.get("execution_result") is not None:
                    print("\n--- Resultado do comando ---")
                    print(json.dumps(result["execution_result"], ensure_ascii=False, indent=4))
                print("----------------------------------------------")

    except Exception as e:
        print(f"\n[ERRO FATAL] O orquestrador parou. Motivo: {e}")