# Start Simpleem Online

## Step 1 — Kill old sessions first

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null; lsof -ti:5173 | xargs kill -9 2>/dev/null; echo "Killed"
```

## Step 2 — Backend (port 8000)

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-online" && ~/.pyenv/versions/3.10.14/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Step 3 — Frontend (port 5173) (new terminal)

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-online/frontend" && npm run dev
```

## Step 4 — Open in browser

```
http://localhost:5173
```
