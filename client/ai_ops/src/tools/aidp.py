# aidp_query_tool.py
import os
import contextlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from src.common.config import *

# You need: pip install jaydebeapi
import jaydebeapi


class AIDPQueryInput(BaseModel):
    sql: str = Field(..., description="SQL to run against AIDP (Spark SQL / SQL-92).")
    catalog: Optional[str] = Field(None, description="Optional catalog name.")
    schema: Optional[str] = Field(None, description="Optional schema (database) name.")
    max_rows: int = Field(1000, description="Max rows to return.")


class AIDPQueryTool(BaseTool):
    """
    LangChain Tool to query OCI AIDP via Simba Spark JDBC.

    Auth modes supported (see AIDP docs):
      - API key / OCI profile (recommended for servers/headless)
      - Authorization token (browser SSO; not ideal for headless)

    Required env:
      AIDP_JDBC_URL  -> Full JDBC URL from AIDP 'Connection details'
      AIDP_JDBC_JAR  -> Path to Spark JDBC jar (e.g., SparkJDBC42.jar)

    Optional env:
      OCI_CONFIG_FILE -> Non-default path to OCI config file
      OCI_PROFILE     -> Non-default OCI profile name
    """
    name: str = "aidp_sql_query"
    description: str = (
        "Run SQL against OCI AIDP (Spark) over JDBC and return rows as JSON-safe dicts."
    )

    def _connect(self):
        jdbc_url = AIDP_JDBC_URL.strip()
        jar_path = AIDP_JDBC_JAR.strip()
        if not jdbc_url or not jar_path:
            raise RuntimeError(
                "AIDP_JDBC_URL and AIDP_JDBC_JAR must be set in the environment."
            )

        # If custom config/profile specified, append to JDBC URL (per AIDP doc)
        # You can also bake these into AIDP_JDBC_URL directly; this is just convenience.
        oci_cfg = AIDP_OCI_CONFIG_FILE
        oci_prof = AIDP_OCI_PROFILE
        if oci_cfg and "OCIConfigFile=" not in jdbc_url:
            sep = ";" if not jdbc_url.endswith(";") else ""
            jdbc_url = f"{jdbc_url}{sep}OCIConfigFile={oci_cfg}"
        if oci_prof and "OCIProfile=" not in jdbc_url:
            sep = ";" if not jdbc_url.endswith(";") else ""
            jdbc_url = f"{jdbc_url}{sep}OCIProfile={oci_prof}"

        # Simba driver class from AIDP doc/instructions
        driver_class = "com.simba.spark.jdbc.Driver"

        # JayDeBeApi params: (driver, url, [user, pass?], jars)
        # For AIDP, user/pass are not used with OCI profile/token auth.
        conn = jaydebeapi.connect(driver_class, jdbc_url, [], jar_path)
        print(f"connections :  {conn}")
        return conn

    def _run(self, sql: str, catalog: Optional[str] = None,
             schema: Optional[str] = None, max_rows: int = 1000) -> Dict[str, Any]:
        # Build a safe preamble for catalog/schema if provided.
        preamble = []
        # Spark supports catalog+schema use statements in recent versions.
        if catalog:
            preamble.append(f"USE CATALOG `{catalog}`")
        if schema:
            preamble.append(f"USE `{schema}`")

        # Enforce a limit on the client side too (Spark also supports LIMIT in SQL if you want).
        result: Dict[str, Any] = {"rows": [], "rowcount": 0}
        with contextlib.closing(self._connect()) as conn:
            with contextlib.closing(conn.cursor()) as cur:
                for stmt in preamble:
                    cur.execute(stmt)
                cur.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                count = 0
                while True:
                    row = cur.fetchone()
                    if row is None or count >= max_rows:
                        break
                    result["rows"].append({cols[i]: row[i] for i in range(len(cols))})
                    count += 1
                result["rowcount"] = count
        return result

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        # Simple sync wrapper; if you need true async, run in thread executor.
        return self._run(*args, **kwargs)


def make_structured_tool() -> StructuredTool:
    """
    Convenience for plugging into LangChain/LangGraph ReAct agents.
    """
    tool = AIDPQueryTool()
    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=AIDPQueryInput,
        func=lambda sql, catalog=None, schema=None, max_rows=1000: tool._run(
            sql=sql, catalog=catalog, schema=schema, max_rows=max_rows
        ),
    )

if __name__ == "__main__":
    make_structured_tool()
