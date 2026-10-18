"""
Launcher: starts app.py as a fully detached subprocess and writes
PID + startup confirmation to server.log
"""
import subprocess, sys, os, time

wdir   = os.path.dirname(os.path.abspath(__file__))
log    = os.path.join(wdir, "server.log")
pyexe  = sys.executable

env = os.environ.copy()
env["PYTHONUTF8"] = "1"

with open(log, "w") as lf:
    proc = subprocess.Popen(
        [pyexe, "app.py"],
        cwd=wdir,
        stdout=lf,
        stderr=lf,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        if sys.platform == "win32" else 0,
    )

with open(os.path.join(wdir, "server.pid"), "w") as pf:
    pf.write(str(proc.pid))

print(f"Flask server started — PID {proc.pid}")
print(f"Log file: {log}")

# Poll until Flask says it's running (max 40 s)
for i in range(40):
    time.sleep(1)
    try:
        content = open(log).read()
    except Exception:
        content = ""
    if "Running on http" in content:
        print(f"  Server is UP after {i+1}s")
        print("--- server.log ---")
        print(content)
        break
    if proc.poll() is not None:
        print(f"  Process exited with code {proc.returncode}")
        print("--- server.log ---")
        try:
            print(open(log).read())
        except Exception:
            pass
        sys.exit(1)
    print(f"  waiting... ({i+1}s)")
else:
    print("Timed out waiting for Flask to start. Log:")
    try:
        print(open(log).read())
    except Exception:
        pass
