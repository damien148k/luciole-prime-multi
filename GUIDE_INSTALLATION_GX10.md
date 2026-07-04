# Guide d'installation — Luciole Prime Multi (GX10)

> **Machine cible** : NVIDIA DGX Spark GX10 — Grace Blackwell GB10, arm64, sm_121  
> **Utilisateur terminal** : `dam@gx10-ca25`  
> **Repo** : `luciole-prime-multi` (privé)  
> **Objectif** : Déployer Luciole multi-instances avec LLM TRT-LLM partagé

---

## Sommaire

1. [Premier démarrage et mise à jour du GX10](#1-premier-démarrage-et-mise-à-jour-du-gx10)
2. [Docker — permissions et groupe](#2-docker--permissions-et-groupe)
3. [NVIDIA Container Runtime — configuration Docker](#3-nvidia-container-runtime--configuration-docker)
4. [Compte NGC personnel et API Key](#4-compte-ngc-personnel-et-api-key)
5. [Login Docker sur nvcr.io](#5-login-docker-sur-nvcrIo)
6. [Cloner le repo luciole-prime-multi](#6-cloner-le-repo-luciole-prime-multi)
7. [Téléchargement des embeddings (venv Python)](#7-téléchargement-des-embeddings-venv-python)
8. [Téléchargement du modèle LLM (Qwen3-30B-A3B-NVFP4)](#8-téléchargement-du-modèle-llm-qwen3-30b-a3b-nvfp4)
9. [Build de l'image Docker GPU arm64](#9-build-de-limage-docker-gpu-arm64)
10. [Démarrage du stack LLM partagé](#10-démarrage-du-stack-llm-partagé)
11. [Installation d'une instance métier](#11-installation-dune-instance-métier)
12. [Vérification finale](#12-vérification-finale)
13. [Gestion des instances](#13-gestion-des-instances)
14. [Erreurs courantes et solutions](#14-erreurs-courantes-et-solutions)

---

## 1. Premier démarrage et mise à jour du GX10

Au premier démarrage, mettre à jour le système avant toute manipulation Docker ou NGC.

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo reboot
```

Après redémarrage, vérifier l'architecture et la version du système :

```bash
uname -m          # doit afficher : aarch64
cat /etc/os-release
```

---

## 2. Docker — permissions et groupe

### Problème rencontré

```
permission denied while trying to connect to the Docker API at unix:///var/run/docker.sock
```

> **Cause** : l'utilisateur `dam` n'appartient pas au groupe `docker`. Le `chmod` sur un fichier `.tar` ne change rien à ce problème.

### Solution (permanente)

```bash
# Ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER

# Appliquer sans logout (session courante uniquement)
newgrp docker

# Vérifier
docker ps
```

> **Important** : `newgrp docker` ne vaut que pour la session courante.  
> Pour que ce soit permanent, déconnecter/reconnecter ou redémarrer la machine.  
> En attendant, préfixer toutes les commandes Docker avec `sudo`.

### Vérification des permissions du socket

```bash
ls -l /var/run/docker.sock
# Attendu : srw-rw---- 1 root docker ...
```

Si les permissions sont incorrectes :

```bash
sudo chown root:docker /var/run/docker.sock
sudo chmod 660 /var/run/docker.sock
sudo systemctl restart docker
```

---

## 3. NVIDIA Container Runtime — configuration Docker

### Problème rencontré

```
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```

> **Cause** : le runtime Docker par défaut est `runc` au lieu de `nvidia`.  
> `NVIDIA_VISIBLE_DEVICES=all` dans les variables d'environnement n'a aucun effet sans le runtime NVIDIA.  
> `DeviceRequests: null` dans `docker inspect` confirme que le GPU n'est pas injecté.

### Vérification

```bash
sudo docker info | grep -i runtime
# Si "Default Runtime: runc" → appliquer la correction ci-dessous

which nvidia-container-runtime
# Doit retourner : /usr/bin/nvidia-container-runtime
# (installé avec le NVIDIA Container Toolkit)
```

### Solution

```bash
# Créer le daemon.json Docker avec nvidia comme runtime par défaut
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

# Redémarrer Docker
sudo systemctl restart docker

# Vérifier
sudo docker info | grep -i runtime
# Attendu : Default Runtime: nvidia
```

> **Note** : cette étape est **obligatoire avant tout `docker compose up`** impliquant le GPU.  
> Sans elle, `NVIDIA_VISIBLE_DEVICES=all` est ignoré et `libcuda.so.1` est introuvable.

---

## 4. Compte NGC personnel et API Key

### Problème rencontré

Sur le portail NGC, lors de la génération d'une API key, le bandeau suivant apparaît :

```
API Access Restricted by your Organization.
```

> **Cause** : le compte NVIDIA/NGC est rattaché à une organisation (compte pro ou entreprise)  
> qui bloque la génération d'API key pour les utilisateurs individuels.

### Solution

Créer un compte NGC **personnel** séparé, non rattaché à une organisation.

1. Aller sur [https://ngc.nvidia.com](https://ngc.nvidia.com)
2. Se connecter ou créer un **nouveau compte personnel** avec une adresse email personnelle (ex: Gmail, etc.)
3. S'assurer que ce compte n'est lié à **aucune organisation**
4. Dans le menu du compte : **Setup → Generate API Key → + Generate Personal Key**
5. Copier la clé générée et la stocker dans un gestionnaire de mots de passe — **elle ne s'affiche qu'une seule fois**

---

## 4. Login Docker sur nvcr.io

### Problème rencontré

```
$ docker login nvcr.io
Username: contact@148kprod.com
Password: ****
Get "https://nvcr.io/v2/": unauthorized
```

> **Cause** : le registre NGC n'accepte pas l'email comme identifiant Docker.  
> Il faut obligatoirement utiliser le token technique `$oauthtoken`.

### Procédure correcte

```bash
# Le username est LITTÉRALEMENT "$oauthtoken" — ne pas le remplacer
sudo docker login nvcr.io --username '$oauthtoken'
# Password: <COLLER_LA_CLE_API_NGC_PERSONNELLE>
```

> **Attention** : utiliser des guillemets simples autour de `$oauthtoken` pour  
> éviter que le shell l'interprète comme une variable vide.

Alternativement, avec la clé en variable d'environnement :

```bash
export NGC_API_KEY="<TA_CLE_API_NGC>"
echo "$NGC_API_KEY" | sudo docker login nvcr.io \
  --username '$oauthtoken' \
  --password-stdin
```

Le résultat attendu : `Login Succeeded`

> **Note** : si `docker login` réussit sans `sudo` mais échoue avec `sudo`, c'est que  
> les credentials sont stockés dans le contexte de l'utilisateur, pas de root.  
> Utiliser systématiquement `sudo docker login` pour que `sudo docker pull` fonctionne.

---

## 5. Cloner le repo luciole-prime-multi

```bash
cd ~/Documents
git clone git@github.com:damien148k/luciole-prime-multi.git
cd luciole-prime-multi
```

Structure du repo :

```
luciole-prime-multi/
├── docker-compose.shared-llm.yml       # Stack LLM partagé (base)
├── docker-compose.shared-llm.gx10.yml  # Override GPU GX10
├── docker-compose.instance.yml         # Stack instance métier (base)
├── docker-compose.instance.gx10.yml    # Override GPU par instance
├── Dockerfile.gpu.arm64                # Image RAG arm64
├── rag-system/                         # Code Python (agent, chat, admin, watcher, mail)
├── config/                             # Configuration YAML
├── scripts/
│   ├── install_gx10.sh                 # Installeur interactif (lancer en premier)
│   ├── stop_instance.sh                # Arrêter une instance
│   ├── list_instances.sh               # Lister toutes les instances
│   ├── trt_entrypoint.gx10.sh         # Entrypoint TRT-LLM GX10
│   ├── prepare_gx10.sh                 # Prépare dossiers + CUTLASS config
│   └── download_model.sh               # Télécharge le modèle LLM
├── instances/                          # Créé par install_gx10.sh (une sous-dossier par métier)
│   └── <metier>/
│       ├── .env
│       ├── data/
│       ├── config/
│       └── models/ -> ../../models/    # Lien symbolique vers embeddings partagés
├── models/
│   └── huggingface/                    # Embeddings partagés entre toutes les instances
├── .env.template                       # Template de variables d'environnement
└── README.md
```

---

## 6. Téléchargement des embeddings (venv Python)

### Problème rencontré

```
error: externally-managed-environment
× This environment is externally managed
```

> **Cause** : sur Debian/Ubuntu récents (PEP 668), `pip install` en système est bloqué  
> pour éviter de casser les paquets `apt`. Lancer `sudo bash download_model.sh` sans venv active  
> ce blocage, même avec `sudo`.

### Solution : utiliser le venv dédié

Le script `prepare_gx10.sh` gère la création du venv et le téléchargement des embeddings.  
Ne **pas** utiliser `sudo pip install` ni `pip install --break-system-packages`.

```bash
cd ~/Documents/luciole-prime-multi

# Exécuter avec sudo bash (chemin explicite obligatoire — sudo ne trouve pas les scripts dans $PWD)
sudo bash scripts/prepare_gx10.sh
```

> **Erreur classique** : `sudo prepare_gx10.sh` → `commande introuvable`  
> **Toujours** utiliser `sudo bash scripts/<nom_du_script>.sh`

Ce script :
- Crée le dossier `models/huggingface/` pour les embeddings partagés
- Crée un venv Python dans `~/luciole-venv/` si absent
- Télécharge `bge-m3` et `bge-reranker-v2-m3` depuis HuggingFace
- Configure le fichier `extra-llm-api-config.yml` avec `moe_config: backend: CUTLASS`
- Ajuste les permissions (`chown`)

---

## 7. Téléchargement du modèle LLM (Qwen3-30B-A3B-NVFP4)

Le modèle est téléchargé depuis HuggingFace dans `models/hf_models/`.  
Il est **partagé entre toutes les instances** — ne télécharger qu'une seule fois.

```bash
cd ~/Documents/luciole-prime-multi

# Activer le venv créé par prepare_gx10.sh
source ~/luciole-venv/bin/activate

# Télécharger le modèle
bash scripts/download_model.sh
```

> Le modèle Qwen3-30B-A3B-Instruct-2507-NVFP4 est volumineux (~15-20 Go).  
> Le téléchargement peut prendre 30 à 60 minutes selon la connexion.

---

## 8. Build de l'image Docker GPU arm64

L'image `luciole-gpu:arm64` est buildée nativement sur le GX10.  
Elle est partagée par toutes les instances (qdrant, opensearch, agent, chat, admin, feedback, watcher, mail sont des images standard — seul `luciole-gpu:arm64` est buildé localement).

```bash
cd ~/Documents/luciole-prime-multi

# Build (prend 10-20 minutes au premier build)
sudo docker build -f Dockerfile.gpu.arm64 -t luciole-gpu:arm64 .
```

> **Note rebuild** : pour forcer un rebuild complet sans cache :  
> `sudo docker build --no-cache -f Dockerfile.gpu.arm64 -t luciole-gpu:arm64 .`

---

## 9. Démarrage du stack LLM partagé

Le LLM TRT-LLM est démarré **une seule fois** et reste actif pour toutes les instances.  
Ne pas le redémarrer entre les installations d'instances métier.

```bash
cd ~/Documents/luciole-prime-multi

sudo docker compose \
  -f docker-compose.shared-llm.yml \
  -f docker-compose.shared-llm.gx10.yml \
  up -d
```

Vérifier que le container est en cours de démarrage :

```bash
sudo docker ps | grep tensorrt
# luciole-tensorrt-shared   starting (le démarrage prend 5-10 minutes)
```

Surveiller les logs :

```bash
sudo docker logs -f luciole-tensorrt-shared
```

Le modèle est prêt quand les logs affichent :

```
Starting Triton Server...
I ... Started GRPCInferenceService...
I ... Started HTTPService...
```

Vérification via l'API :

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
# Doit retourner la liste des modèles disponibles
```

> **Note** : le healthcheck est configuré avec `start_period: 600s` — Docker affichera  
> `starting` pendant jusqu'à 10 minutes, c'est normal.

---

## 10. Installation d'une instance métier

Une fois le stack LLM partagé démarré et healthy, installer la première instance métier.

```bash
cd ~/Documents/luciole-prime-multi

sudo bash scripts/install_gx10.sh
```

Le script est interactif :

```
════════════════════════════════════════════════════════════
   Luciole — Installation d'une nouvelle instance métier
════════════════════════════════════════════════════════════

Pour quel métier / client ? (ex: support, juridique, chavenay) : chavenay

[INFO] Ports assignés à l'instance 'chavenay' :
   API (agent)    : 8000
   Admin UI       : 8001
   Chat UI        : 8002
   Feedback UI    : 8003
   Qdrant         : 8004
   OpenSearch     : 8005
   Watcher        : 8006
   Mail SMTP      : 8007
   Mail IMAP      : 8008
   Mail Admin     : 8009

Confirmer l'installation ? (O/n) : O
```

Le script :
1. Valide que le réseau `luciole_shared` existe (LLM partagé actif)
2. Détecte automatiquement les ports libres (blocs de 10 par instance)
3. Crée `instances/chavenay/` avec les sous-dossiers data, config, feedbacks, backups
4. Crée un lien symbolique vers les embeddings partagés (`models/huggingface/`)
5. Génère `instances/chavenay/.env` avec tous les ports et la clé de chiffrement mail
6. Lance le stack Docker de l'instance

Accès après installation :

```
Chat UI    → http://localhost:8002
Admin UI   → http://localhost:8001
Feedback   → http://localhost:8003
API        → http://localhost:8000
```

### Installation d'une deuxième instance

Relancer le script — il détecte automatiquement les ports déjà utilisés et assigne le bloc suivant :

```bash
sudo bash scripts/install_gx10.sh
# Répondre : juridique
# Ports auto-assignés : 8010-8019
```

---

## 11. Vérification finale

### Tous les containers

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

Résultat attendu :

```
NAMES                           STATUS                    PORTS
luciole-tensorrt-shared         Up X minutes (healthy)    127.0.0.1:8001->8000/tcp
luciole-qdrant-chavenay         Up X minutes              127.0.0.1:8004->6333/tcp
luciole-opensearch-chavenay     Up X minutes              127.0.0.1:8005->9200/tcp
luciole-agent-chavenay          Up X minutes              0.0.0.0:8000->8000/tcp
luciole-admin-chavenay          Up X minutes              0.0.0.0:8001->8080/tcp
luciole-chat-chavenay           Up X minutes              0.0.0.0:8002->8501/tcp
luciole-feedback-chavenay       Up X minutes              0.0.0.0:8003->8503/tcp
luciole-watcher-chavenay        Up X minutes              127.0.0.1:8006->8090/tcp
luciole-mail-chavenay           Up X minutes (healthy)    ...
```

### Test du LLM partagé

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-30b-a3b-instruct","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":50}' \
  | python3 -m json.tool
```

### Test de l'API RAG

```bash
curl -s http://localhost:8000/health
# {"status": "ok", "instance": "chavenay"}
```

### Test du watcher

```bash
curl -s http://localhost:8006/status
```

### Création du compte GreenMail (service mail)

Après le démarrage des containers, créer le compte mail local utilisé par Luciole :

```bash
# Remplacer <INSTANCE_NAME> par le nom de l'instance (ex: support, chavenay...)
curl -s -X POST http://localhost:${GREENMAIL_ADMIN_PORT:-8019}/api/user \
  -H "Content-Type: application/json" \
  -d '{"email":"luciole@luciole.local","login":"luciole","password":"luciole"}'
# Réponse attendue : {"login":"luciole","email":"luciole@luciole.local"}
```

Puis configurer le client mail externe (Outlook, Thunderbird...) :

| Paramètre | Valeur |
|---|---|
| Email | `luciole@luciole.local` |
| Login | `luciole` |
| Mot de passe | `luciole` |
| IMAP host | `<IP_GX10>` |
| IMAP port | `${MAIL_IMAP_PORT:-8018}` — **sans chiffrement** |
| SMTP host | `<IP_GX10>` |
| SMTP port | `${MAIL_SMTP_PORT:-8017}` — **sans chiffrement** |

> **Note** : le compte GreenMail est recréé à chaque suppression du container `luciole-mail-<instance>`. À refaire après un `docker rm`.

### Ingestion d'un document test

Copier un fichier PDF ou DOCX dans `instances/chavenay/data/chavenay/` :

```bash
cp /path/to/document.pdf instances/chavenay/data/chavenay/
# Le watcher détecte le fichier et lance l'indexation automatiquement
```

Vérifier l'indexation dans les logs du watcher :

```bash
sudo docker logs -f luciole-watcher-chavenay
```

---

## 12. Gestion des instances

### Lister les instances

```bash
sudo bash scripts/list_instances.sh
```

### Arrêter une instance

```bash
sudo bash scripts/stop_instance.sh chavenay
```

### Arrêter le stack LLM partagé

> **Attention** : arrêter le LLM partagé interrompt **toutes** les instances actives.

```bash
cd ~/Documents/luciole-prime-multi
sudo docker compose \
  -f docker-compose.shared-llm.yml \
  -f docker-compose.shared-llm.gx10.yml \
  down
```

### Redémarrer une instance (rebuild image)

```bash
# Toujours stop + rm + up — ne pas utiliser --force-recreate seul (garde l'ancienne image)
# IMPORTANT : toujours inclure --profile gpu sinon les services GPU démarrent sur un réseau séparé
sudo docker stop luciole-agent-chavenay && sudo docker rm luciole-agent-chavenay
cd instances/chavenay
sudo docker compose -f docker-compose.yml -f docker-compose.gx10.yml \
  --project-name luciole-chavenay --profile gpu up -d
```

### Mettre à jour le code (git pull + rebuild)

```bash
cd ~/Documents/luciole-prime-multi
git pull

# Rebuild image
sudo docker build --no-cache -f Dockerfile.gpu.arm64 -t luciole-gpu:arm64 .

# Recréer les containers de chaque instance
sudo bash scripts/stop_instance.sh chavenay
sudo bash scripts/install_gx10.sh  # choisir 'chavenay' → reconfigurer
```

---

## 13. Erreurs courantes et solutions

### `permission denied while trying to connect to the Docker API`

```bash
sudo usermod -aG docker $USER
newgrp docker          # ou redémarrer la machine
```

### `sudo: download_model.sh: commande introuvable`

`sudo` ne cherche pas dans le répertoire courant.  
Toujours utiliser `sudo bash scripts/<nom>.sh` avec le chemin explicite.

### `Get "https://nvcr.io/v2/": unauthorized`

Deux causes possibles :
1. Mauvais couple username/password → utiliser `$oauthtoken` (littéral) comme username
2. Compte NGC rattaché à une organisation qui bloque les API keys → créer un compte NGC personnel

### `error: externally-managed-environment` (pip)

Ne pas utiliser `sudo pip install`. Utiliser le venv :

```bash
source ~/luciole-venv/bin/activate
pip install <package>
```

### `--force-recreate` garde l'ancienne image

`--force-recreate` recrée le container mais réutilise l'image en cache.  
Pour forcer l'utilisation d'une nouvelle image :

```bash
sudo docker stop <container> && sudo docker rm <container>
# puis relancer — TOUJOURS avec --profile gpu
cd instances/<metier>
sudo docker compose -f docker-compose.yml -f docker-compose.gx10.yml \
  --project-name luciole-<metier> --profile gpu up -d
```

### Le LLM partagé répond `connection refused` depuis une instance

Vérifier que le container `luciole-tensorrt-shared` est sur le réseau `luciole_shared` :

```bash
sudo docker network inspect luciole_shared
```

Le container de l'instance doit aussi être sur ce réseau — vérifier dans `docker-compose.instance.yml`.

### `_resolve_index_name: 'chavenay' reçu mais ignoré — force 'luciole'`

Variable `MULTI_INDEX_MODE=true` manquante dans le `.env` de l'instance.  
Le fichier `.env` généré par `install_gx10.sh` ne contient pas encore cette variable.  
L'ajouter manuellement :

```bash
echo "MULTI_INDEX_MODE=true" >> instances/chavenay/.env
# puis redémarrer l'instance (stop/rm/up)
```

### Healthcheck mail en échec (`nc` ou `curl` absents dans l'image GreenMail)

Le healthcheck utilise `/dev/tcp` (bash built-in) :

```bash
bash -c 'echo > /dev/tcp/localhost/3025'
```

---

## Récapitulatif — Ordre d'installation

```
1.  sudo apt update && upgrade → reboot
2.  sudo usermod -aG docker $USER → newgrp docker
3.  Configurer NVIDIA Container Runtime → sudo tee /etc/docker/daemon.json → sudo systemctl restart docker
4.  Créer compte NGC personnel → générer API key
5.  sudo docker login nvcr.io --username '$oauthtoken'
6.  git clone luciole-prime-multi
7.  sudo bash scripts/prepare_gx10.sh    ← embeddings + CUTLASS config
8.  source ~/luciole-venv/bin/activate && bash scripts/download_model.sh  ← modèle LLM
9.  sudo docker build -f Dockerfile.gpu.arm64 -t luciole-gpu:arm64 .
10. sudo docker compose -f docker-compose.shared-llm.yml -f docker-compose.shared-llm.gx10.yml up -d
11. (attendre que tensorrt-shared soit healthy)
12. sudo bash scripts/install_gx10.sh  ← première instance métier
13. Vérification : docker ps, curl /health, ingestion document test
```

---

*Guide généré le 2026-07-04 — basé sur les difficultés réelles rencontrées lors de l'installation initiale du GX10 (dam@gx10-ca25).*
