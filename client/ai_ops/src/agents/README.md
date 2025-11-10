# DEPLOYMENT OF A LOCAL ORACLE DB

### 1) Is the VM even running?
podman machine list

### 2) Start (or restart) it
podman machine stop || true
podman machine start

### 3) Use Podman’s managed connection instead of a hardcoded port
podman system connection ls
podman system connection default  podman-machine-default-root  # pick the right name from the list

### 4) Test
podman info
podman ps

### 5) (Official Oracle Database Free container image.) 
podman pull container-registry.oracle.com/database/free:latest

### 6)
podman volume create oradata

### ) Optional Step

1. Check what’s running / existing
podman ps -a | grep oracle-free

That will show if the container is running or just stopped.

2. If you want to reuse the same name

Stop it:

podman stop oracle-free


Remove it:

podman rm oracle-free


Then re-run your podman run ... --name oracle-free .... Step 7

### 7) 
podman run -d --name oracle-free \
  -p 1521:1521 \
  -v oradata:/opt/oracle/oradata \
  -e ORACLE_PWD=Oracle_123 \
  container-registry.oracle.com/database/free:latest

### 8) 
podman logs -f oracle-free   # watch startup, this may take longer time

### 9) Connect
#### Default services: FREE (CDB) and FREEPDB1 (PDB). Example with SQL*Plus/SQLcl:

sql sys/Oracle_123@localhost:1521/FREEPDB1 as sysdba

### or inside the container
podman exec -it oracle-free bash
sql sys/Oracle_123@localhost:1521/FREEPDB1 as sysdba

### 10 ) Final step to enable MCP 
conn -save demo_mcp_sys -savepwd sys/Oracle_123@localhost:1521/FREEPDB1 as sysdba

--- ~/.sqlcl/connections.json ---

# Connect to ADB using wallet
> 
  sql /nolog

  set cloudconfig /etc/ords/config/Wallet_<dbname>.zip 

  show tns #if it is from wallet

  conn user/pwd@db_high

  conn -save myconn -savepwd

  CONNMGR LIST

  cm list

  cm show myconn

  connmgr test myconn

  CONNECT myconn

  show connection

  show user

  show user con_name

  conn -n conn_name
  
  CONNMGR DELETE -conn myconn

  #### STREAMLIT

  export PYTHONPATH=/Users/aojah/Documents/GenAI-CoE/Agentic-Framework/source-code/agentic-ai-landing-zone-master/agentic-ai-landing-zone/client/ai_ops
