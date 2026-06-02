# TARGET2 GPT

Instancia independiente tipo GPT para TARGET Services, centrada en T2, CLM y RTGS.

No usa `tips/`, `t2sgpt/` ni `trilemma-simulator`. Su corpus vive en `target2gpt/data/`, su salida en `target2gpt/output/` y sus secretos en `target2gpt/secrets/`.

## Flujo rapido

```powershell
cd C:\Users\Jose-Firebat\proyectos\trilemma\target2gpt
python -m pip install -r requirements.txt
python target2_ingest.py --force
python qa_target2.py
python target2_web.py
```

La web local queda en `http://127.0.0.1:8791/target2/`, salvo que cambies `TARGET2_WEB_PORT`.

## Ingesta

`target2_ingest.py` arranca en la raiz oficial de ECB Professional Use:

```text
https://www.ecb.europa.eu/paym/target/target-professional-use-documents-links/html/index.en.html
```

Desde ahi sigue las paginas profesionales en ingles de Shared features, T2, T2S, TIPS, ECMS y Pontes, y descarga los documentos de texto enlazados: PDF, HTML, XLSX, XML, XSD, ZIP y formatos equivalentes. El objetivo es tener todo Professional Use disponible como corpus local, con CLM/RTGS y documentacion comun priorizados por el motor de respuesta.

Para una prueba corta:

```powershell
python target2_ingest.py --limit 20 --max-pages 12
```

Para reconstruir completo:

```powershell
python target2_ingest.py --force --max-pages 512
```

## Web y acceso

Variables principales:

```powershell
$env:TARGET2_WEB_HOST="0.0.0.0"
$env:TARGET2_WEB_PORT="8791"
$env:TARGET2_AUTH_DISABLED="true"
$env:TARGET2_PUBLIC_BASE_URL="https://target2-api.ethcuela.es"
$env:TARGET2_SESSION_SECRET="pon-un-secreto-largo-aleatorio"
```

Gestion de usuarios:

```powershell
python access_admin.py approve usuario@example.com --name "Usuario"
python access_admin.py list
```

## Watcher

```powershell
python target2_doc_watcher.py --check-only --verbose
```

## Publicacion

Repositorio previsto:

```text
https://github.com/jsmcel/target2gpt
```

El repo contiene codigo y configuracion. `data/`, `output/` y `secrets/` estan ignorados por Git.
