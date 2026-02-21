# Start Simpleem Offline

## Step 1 — Kill old sessions first

```bash
lsof -ti:9000 | xargs kill -9 2>/dev/null; lsof -ti:4000 | xargs kill -9 2>/dev/null; echo "Killed"
```

## Step 2 — Backend (port 9000)

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline" && ~/.pyenv/versions/3.10.14/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 9000 --reload
```

## Step 3 — Frontend (port 4000) (new terminal)

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline/frontend" && npm run dev
```

## Step 4 — Open in browser

```
http://localhost:4000
```
