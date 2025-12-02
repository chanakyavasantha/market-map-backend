# How to Restart the Backend Server

The server needs to be restarted to pick up route changes. Here's how:

## Quick Restart

1. **Find the terminal where the server is running**
   - Look for the terminal showing uvicorn output

2. **Stop the server**
   - Press `Ctrl+C` in that terminal

3. **Restart the server**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Or Use the Start Script

```bash
cd backend
./start_server.sh
```

## Verify It's Working

After restarting, test these endpoints:

```bash
# Test root
curl http://localhost:8000/

# Test indices status
curl http://localhost:8000/api/indices/status

# Test locations
curl http://localhost:8000/api/indices/locations
```

You should get JSON responses, not "Not Found" errors.

## Check What Port the Server is On

If you're not sure which port the server is using:

```bash
# Check what's running on port 8000
lsof -i :8000

# Or check all uvicorn processes
ps aux | grep uvicorn
```

## If Port is Already in Use

If you get "port already in use" error:

1. Find the process:
   ```bash
   lsof -i :8000
   ```

2. Kill it:
   ```bash
   kill -9 <PID>
   ```

3. Restart the server

