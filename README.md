# Identificação de sotaques regionais brasileiros

Nesse repositório, você encontrará uma base de código para treinamento de modelos de redes neurais para identificação de sotaques regionais, aqui aplicados ao contexto brasileiro.

## Como funciona

(descrever melhor)

- JSON configs and models
- TensorFlow
- MinIO
- PostgreSQL

## Como usar

- Criar um arquivo de variáveis de ambiente tal como o seguinte:
```
MINIO_ENDPOINT=minio.server.hostname
MINIO_ACCESS_KEY=minio_access_key
MINIO_SECRET_KEY=minio_secret_key
MINIO_BUCKET=my-bucket
MINIO_PATH_PREFIX=path_prefix/
DB_CONNECTION_URL=postgresql://user:password@hostname/database
```

## Base de dados

Aguardar.