# Luciole Prime — Multi-instances GX10

Version multi-instances de Luciole pour le GX10 (DGX Spark / GB10 Grace Blackwell, arm64).

**Architecture** : un LLM TRT-LLM partagé + N instances métier indépendantes (une par client/département).

```
GX10
├── Stack LLM partagé (luciole-tensorrt-shared)   ← un seul, toujours actif
│   └── Qwen3-30B-A3B-Instruct-2507-NVFP4
│
├── Instance : support                             ← Chat :8002, Admin :8003
│   ├── luciole-agent-support
│   ├── luciole-qdrant-support
│   ├── luciole-watcher-support
│   └── ...
│
├── Instance : juridique                           ← Chat :8012, Admin :8013
│   ├── luciole-agent-juridique
│   └── ...
│
└── Instance : chavenay                            ← Chat :8022, Admin :8023
    └── ...
```

---

## Prérequis

- GX10 (DGX Spark, GB10, arm64, sm_121)
- Docker + Docker Compose v2
- Image `luciole-gpu:arm64` buildée
- Modèle NVFP4 téléchargé dans `models/hf_models/`
- Embeddings téléchargés dans `models/huggingface/`

---

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/damien148k/luciole-prime-multi
cd luciole-prime-multi
```

### 2. Préparer l'environnement (première fois uniquement)

```bash
# Télécharge les embeddings (bge-m3 + bge-reranker-v2-m3)
bash scripts/download_embeddings.sh

# Télécharge le modèle NVFP4
bash scripts/download_model.sh

# Build l'image applicative
sudo docker build -f Dockerfile.gpu.arm64 -t luciole-gpu:arm64 .
```

### 3. Démarrer le LLM partagé (une seule fois)

```bash
sudo docker compose \
  -f docker-compose.shared-llm.yml \
  -f docker-compose.shared-llm.gx10.yml \
  up -d
```

Attends que le LLM soit healthy (~5-10 min au premier démarrage) :
```bash
sudo docker logs -f luciole-tensorrt-shared 2>&1 | grep "ready\|model"
```

### 4. Installer une instance métier

```bash
sudo bash scripts/install_gx10.sh
```

Le script demande le nom du métier, détecte les ports libres, génère le `.env` et lance les containers.

**Exemple :**
```
Pour quel métier / client ? : chavenay
→ Chat UI  : http://localhost:8000
→ Admin UI : http://localhost:8001
→ API      : http://localhost:8002
```

Répéter pour chaque métier supplémentaire.

---

## Gestion des instances

```bash
# Lister toutes les instances et leur état
bash scripts/list_instances.sh

# Arrêter une instance
sudo bash scripts/stop_instance.sh chavenay

# Redémarrer une instance
cd instances/chavenay
sudo docker compose --project-name luciole-chavenay --profile gpu up -d

# Logs d'une instance
sudo docker compose --project-name luciole-chavenay logs -f agent
```

---

## Structure des dossiers

```
luciole-prime-multi/
├── docker-compose.shared-llm.yml       ← Stack LLM partagé
├── docker-compose.shared-llm.gx10.yml  ← Override GX10
├── docker-compose.instance.yml         ← Template instance métier
├── docker-compose.instance.gx10.yml    ← Override GX10 instance
├── Dockerfile.gpu.arm64                ← Image applicative
├── scripts/
│   ├── install_gx10.sh                 ← Installeur interactif
│   ├── stop_instance.sh                ← Arrêt d'une instance
│   ├── list_instances.sh               ← Liste des instances
│   ├── download_model.sh               ← Téléchargement LLM
│   ├── download_embeddings.sh          ← Téléchargement embeddings
│   ├── prepare_gx10.sh                 ← Préparation initiale GX10
│   └── trt_entrypoint.gx10.sh         ← Entrypoint TRT-LLM
├── models/
│   ├── hf_models/                      ← Modèle LLM (partagé)
│   └── huggingface/                    ← Cache embeddings (partagé)
├── config/                             ← Config par défaut
└── instances/                          ← Données par instance
    ├── .registry                       ← Registre ports/instances
    ├── chavenay/
    │   ├── .env
    │   ├── data/                       ← Documents à indexer
    │   ├── config/
    │   └── feedbacks/
    └── juridique/
        └── ...
```

---

## Notes importantes

- **LLM partagé** : un seul modèle Qwen3-30B chargé en GPU pour toutes les instances. Les requêtes sont batchées.
- **Embeddings partagés** : `models/huggingface/` est monté en lecture seule dans toutes les instances (lien symbolique).
- **Isolation des données** : chaque instance a son propre Qdrant, OpenSearch, dossier `data/`, et collection vectorielle.
- **Ports** : l'installeur assigne automatiquement des blocs de 10 ports par instance (8000-8009, 8010-8019, etc.).
- **Watcher** : surveille `instances/<metier>/data/` et indexe automatiquement les documents ajoutés/supprimés.
