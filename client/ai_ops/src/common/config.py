
import os
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
load_dotenv(PROJECT_ROOT  / "config/.env") # expects OCI_ vars in .env

#────────────────────────────────────────────────────────
# OCI Security configuration
# ────────────────────────────────────────────────────────
AUTH_TYPE = os.getenv("AUTH_TYPE")
OCI_PROFILE = os.getenv("CONFIG_PROFILE")
#────────────────────────────────────────────────────────
# OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_GENAI_MODEL_ID")
PROVIDER       = os.getenv("PROVIDER")

#────────────────────────────────────────────────────────
# OCI GenAI Agents endpoint configuration
# ────────────────────────────────────────────────────────
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")

# ────────────────────────────────────────────────────────────────
# Configure LangGraph Dev Server end point
# ────────────────────────────────────────────────────────────────

EMBDDING_MODEL_ID = os.getenv("OCI_EMBEDDING_MODEL")

# ────────────────────────────────────────────────────────────────
# Configure MCP Servers
# ────────────────────────────────────────────────────────────────
SQLCLI_MCP_PROFILE = os.getenv("SQLCLI_MCP_PROFILE")
TAVILY_MCP_SERVER = os.getenv("TAVILY_MCP_SERVER")
FILE_SYSTEM_ACCESS_KEY=os.getenv("FILE_SYSTEM_ACCESS_KEY")

# ────────────────────────────────────────────────────────────────
# Configure MCP Connections to SSE (Streamable HTTP)
# ────────────────────────────────────────────────────────────────
MCP_TRANSPORT= os.getenv("MCP_TRANSPORT","stdio") #"stdio" #streamable_http" #stdio" #sse
MCP_SSE_HOST=os.getenv("MCP_SSE_HOST","0.0.0.0")
MCP_SSE_PORT=os.getenv("MCP_SSE_PORT","8001")

# ────────────────────────────────────────────────────────────────
# Configure MCP Connections to SSE (Streamable HTTP) for DB TOOLS
# ────────────────────────────────────────────────────────────────
MCP_TRANSPORT_DBTOOLS= os.getenv("MCP_TRANSPORT_DBTOOLS","stdio") #"stdio" #streamable_http" #stdio" #sse
MCP_SSE_HOST_DBTOOLS=os.getenv("MCP_SSE_HOST_DBTOOLS","0.0.0.0")
MCP_SSE_PORT_DBTOOLS=os.getenv("MCP_SSE_PORT_DBTOOLS","8002")

# ────────────────────────────────────────────────────────────────
# Configure AIDP Connections using JDBC drivers
# ────────────────────────────────────────────────────────────────

AIDP_JDBC_URL=os.getenv("AIDP_JDBC_URL")
AIDP_JDBC_JAR=os.getenv("AIDP_JDBC_JAR")
AIDP_OCI_CONFIG_FILE=os.getenv("AIDP_OCI_CONFIG_FILE")
AIDP_OCI_PROFILE=os.getenv("AIDP_OCI_PROFILE")

