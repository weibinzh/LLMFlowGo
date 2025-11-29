import uuid
from typing import Optional, Dict, Any
from sqlmodel import Field, SQLModel, Column, TEXT, JSON


class ProblemPackage(SQLModel, table=True):
    """
 Define the data model for a 'problem set,' which will map to a table in the database.
    """
    # Use UUID as the primary key to ensure uniqueness
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    
    # The path to store related files on the server
    framework_file_path: str
    data_file_path: str 
    get_instance_file_path: str
    evaluation_file_path: str # Path to the user-defined evaluation file
    
    # The name of the objective function specified by the user for optimization
    target_function_name: Optional[str] = Field(default=None) # This will be updated in a later step
    
    # Used for storing analysis and configuration results
    task_description: Optional[str] = Field(default=None)
    template_program_str: Optional[str] = Field(default=None, sa_column=Column(TEXT))
    # template_file_path: Optional[str] = Field(default=None) # This field is redundant
    
    # LLM configuration for performance optimization
    llm_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
