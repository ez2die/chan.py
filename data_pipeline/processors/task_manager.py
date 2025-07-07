#!/usr/bin/env python3
"""
任务管理器模块
"""

import sys
import os
import threading
import time
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import settings
from logger import data_processor_logger as logger


class TaskStatus(Enum):
    """Task status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class DataTask:
    """Data collection task configuration."""
    id: Optional[int] = None
    exchange: str = "okx"
    symbol: str = ""
    timeframe: str = "1h"
    instrument_type: str = "spot"
    status: str = TaskStatus.ACTIVE.value
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_count: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataTask':
        """Create task from dictionary."""
        # Convert ISO strings back to datetime objects
        datetime_fields = ['created_at', 'updated_at', 'last_run', 'last_success']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class TaskManager:
    """Manager for data collection tasks."""
    
    def __init__(self):
        """Initialize the task manager."""
        self.db_path = Path(settings.data_dir) / "tasks.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exchange TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        instrument_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_run TEXT,
                        last_success TEXT,
                        error_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        UNIQUE(exchange, symbol, timeframe, instrument_type)
                    )
                """)
                conn.commit()
            
            logger.info("Task database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing task database: {e}")
            raise
    
    def add_task(self, task: DataTask) -> int:
        """
        Add a new data collection task.
        
        Args:
            task: DataTask object
            
        Returns:
            Task ID
        """
        try:
            now = datetime.now(timezone.utc)
            task.created_at = now
            task.updated_at = now
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO data_tasks 
                    (exchange, symbol, timeframe, instrument_type, status, 
                     created_at, updated_at, last_run, last_success, error_count, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.exchange, task.symbol, task.timeframe, task.instrument_type,
                    task.status, task.created_at.isoformat(), task.updated_at.isoformat(),
                    task.last_run.isoformat() if task.last_run else None,
                    task.last_success.isoformat() if task.last_success else None,
                    task.error_count, task.error_message
                ))
                task_id = cursor.lastrowid
                conn.commit()
            
            logger.info(f"Added task {task_id}: {task.symbol} {task.timeframe}")
            return task_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Task already exists: {task.symbol} {task.timeframe}")
            return self.get_task_by_config(task.exchange, task.symbol, task.timeframe, task.instrument_type).id
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            raise
    
    def get_task(self, task_id: int) -> Optional[DataTask]:
        """Get a task by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM data_tasks WHERE id = ?", (task_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_task(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None
    
    def get_task_by_config(self, exchange: str, symbol: str, timeframe: str, instrument_type: str) -> Optional[DataTask]:
        """Get a task by configuration."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM data_tasks 
                    WHERE exchange = ? AND symbol = ? AND timeframe = ? AND instrument_type = ?
                """, (exchange, symbol, timeframe, instrument_type))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_task(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting task by config: {e}")
            return None
    
    def get_all_tasks(self, status: Optional[str] = None) -> List[DataTask]:
        """Get all tasks, optionally filtered by status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if status:
                    cursor = conn.execute(
                        "SELECT * FROM data_tasks WHERE status = ? ORDER BY created_at DESC",
                        (status,)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM data_tasks ORDER BY created_at DESC"
                    )
                
                rows = cursor.fetchall()
                return [self._row_to_task(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting all tasks: {e}")
            return []
    
    def get_active_tasks(self) -> List[DataTask]:
        """Get all active tasks."""
        return self.get_all_tasks(TaskStatus.ACTIVE.value)
    
    def update_task(self, task: DataTask):
        """Update an existing task."""
        try:
            task.updated_at = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE data_tasks SET
                        status = ?, updated_at = ?, last_run = ?, last_success = ?,
                        error_count = ?, error_message = ?
                    WHERE id = ?
                """, (
                    task.status, task.updated_at.isoformat(),
                    task.last_run.isoformat() if task.last_run else None,
                    task.last_success.isoformat() if task.last_success else None,
                    task.error_count, task.error_message, task.id
                ))
                conn.commit()
            
            logger.info(f"Updated task {task.id}: {task.symbol} {task.timeframe}")
            
        except Exception as e:
            logger.error(f"Error updating task {task.id}: {e}")
            raise
    
    def delete_task(self, task_id: int):
        """Delete a task."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM data_tasks WHERE id = ?", (task_id,))
                conn.commit()
            
            logger.info(f"Deleted task {task_id}")
            
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            raise
    
    def mark_task_success(self, task_id: int):
        """Mark a task as successfully completed."""
        task = self.get_task(task_id)
        if task:
            now = datetime.now(timezone.utc)
            task.last_run = now
            task.last_success = now
            task.error_count = 0
            task.error_message = None
            self.update_task(task)
    
    def mark_task_error(self, task_id: int, error_message: str):
        """Mark a task as having an error."""
        task = self.get_task(task_id)
        if task:
            now = datetime.now(timezone.utc)
            task.last_run = now
            task.error_count += 1
            task.error_message = error_message
            
            # Mark as error status if too many failures
            if task.error_count >= 3:
                task.status = TaskStatus.ERROR.value
            
            self.update_task(task)
    
    def _row_to_task(self, row: sqlite3.Row) -> DataTask:
        """Convert database row to DataTask object."""
        data = dict(row)
        
        # Convert ISO strings back to datetime objects
        datetime_fields = ['created_at', 'updated_at', 'last_run', 'last_success']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
            else:
                data[field] = None
        
        return DataTask(**data)