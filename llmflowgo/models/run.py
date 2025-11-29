import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlmodel import Field, SQLModel, JSON, Column
from sqlalchemy.types import JSON as JSONB

# Main table model
class Run(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    problem_id: uuid.UUID = Field(foreign_key="problempackage.id", index=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        nullable=False, 
        index=True
    )
    
    status: str = Field(default="pending", index=True)
    
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    
    meoh_config: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    baseline_scores: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))

    # Result fields
    final_population: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB))
    pareto_front: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB))
    best_solution_code: Optional[str] = Field(default=None)
    result_summary: Optional[str] = Field(default=None)
    result_analysis: Optional[str] = Field(default=None)
    
    # 新增：最终结果快照与派生字段
    final_result_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    cloud_count: Optional[int] = Field(default=None)
    edge_count: Optional[int] = Field(default=None)
    device_count: Optional[int] = Field(default=None)
    makespan: Optional[float] = Field(default=None)
    energy: Optional[float] = Field(default=None)
    
    logs: Optional[str] = Field(default=None)

# Model for creating a new run
class RunCreate(SQLModel):
    problem_id: uuid.UUID
    meoh_config: Optional[Dict[str, Any]] = {}

# Model for reading run data (API response)
class RunRead(SQLModel):
    id: uuid.UUID
    problem_id: uuid.UUID
    created_at: datetime
    status: str
    meoh_config: Optional[Dict[str, Any]]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    baseline_scores: Optional[List[Dict[str, Any]]]
    final_population: Optional[List[Dict[str, Any]]]
    pareto_front: Optional[List[Dict[str, Any]]]
    best_solution_code: Optional[str]
    result_summary: Optional[str]
    result_analysis: Optional[str]
    
    # 新增：最终结果与派生字段（读取模型方便前端调试）
    final_result_json: Optional[Dict[str, Any]]
    cloud_count: Optional[int]
    edge_count: Optional[int]
    device_count: Optional[int]
    makespan: Optional[float]
    energy: Optional[float]
    
    logs: Optional[str]
