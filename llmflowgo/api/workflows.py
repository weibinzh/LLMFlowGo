from fastapi import APIRouter, HTTPException
import subprocess
import json
import re
from typing import List, Dict, Any

router = APIRouter()

def run_docker_sql(sql_query: str) -> str:
    """Execute SQL query via Docker"""
    try:
        cmd = [
            "docker", "exec", "postgres", 
            "psql", "-U", "postgres", "-d", "postgres", 
            "-t", "-c", sql_query
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace'  # Handle encoding errors
        )
        
        if result.returncode != 0:
            raise Exception(f"Docker command failed: {result.stderr}")
        
        return result.stdout.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute Docker command: {str(e)}")

def parse_workflow_data(output: str) -> List[Dict[str, Any]]:
    """Parse workflow data from Docker output"""
    workflows = []
    
    if not output.strip():
        return workflows
    
    # Split output by lines
    lines = output.strip().split('\n')
    current_workflow = None
    xml_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new workflow record (starts with a number)
        if re.match(r'^\s*\d+\s*\|', line):
            # Save previous workflow
            if current_workflow is not None and xml_content:
                current_workflow['xml'] = '\n'.join(xml_content).strip()
                workflows.append(current_workflow)
            
            # Start a new workflow
            parts = line.split('|', 1)
            if len(parts) >= 2:
                workflow_id = parts[0].strip()
                xml_start = parts[1].strip()
                
                current_workflow = {
                    'id': int(workflow_id),
                    'xml': ''
                }
                xml_content = [xml_start] if xml_start else []
        else:
            # Continue collecting XML content
            if current_workflow is not None:
                xml_content.append(line)
    
    # Save the last workflow
    if current_workflow is not None and xml_content:
        current_workflow['xml'] = '\n'.join(xml_content).strip()
        workflows.append(current_workflow)
    
    return workflows

@router.get("/workflows/xml")
async def get_workflow_xml():
    """
    Get XML data from the workflow table in the PostgreSQL database
    """
    try:
        # Query XML data using Docker command
        sql_query = "SELECT id, xml FROM workflow WHERE xml IS NOT NULL ORDER BY id LIMIT 10;"
        output = run_docker_sql(sql_query)
        
        # Parse output data
        workflows = parse_workflow_data(output)
        
        return {
            "success": True,
            "message": f"Successfully retrieved {len(workflows)} XML records",
            "xmlData": workflows
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/workflows/xml/{workflow_id}")
async def get_workflow_xml_by_id(workflow_id: int):
    """
    Get specific XML data by ID from the workflow table in the PostgreSQL database
    """
    try:
        # Query specific ID XML data using Docker command
        sql_query = f"SELECT id, xml FROM workflow WHERE id = {workflow_id} AND xml IS NOT NULL;"
        output = run_docker_sql(sql_query)
        
        # Parse output data
        workflows = parse_workflow_data(output)
        
        if not workflows:
            return {
                "success": False,
                "message": f"No XML data found for ID {workflow_id}",
                "xmlData": None
            }
        
        return {
            "success": True,
            "message": f"Successfully retrieved XML data for ID {workflow_id}",
            "xmlData": workflows[0]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/workflows")
async def get_workflows():
    """
    Get basic information for all workflows
    """
    try:
        # Query basic workflow information using Docker command
        sql_query = """
        SELECT id, 
               CASE WHEN xml IS NOT NULL THEN 'true' ELSE 'false' END as has_xml,
               CASE WHEN xml IS NOT NULL THEN LENGTH(xml::text) ELSE 0 END as xml_length
        FROM workflow 
        ORDER BY id;
        """
        output = run_docker_sql(sql_query)
        
        workflows = []
        if output.strip():
            lines = output.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        workflow_id = int(parts[0].strip())
                        has_xml = parts[1].strip().lower() == 'true'
                        xml_length = int(parts[2].strip())
                        
                        workflows.append({
                            "id": workflow_id,
                            "has_xml": has_xml,
                            "xml_length": xml_length
                        })
                    except (ValueError, IndexError):
                        continue
        
        return {
            "success": True,
            "message": f"Successfully retrieved {len(workflows)} workflow records",
            "workflows": workflows
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
