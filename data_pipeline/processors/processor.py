#!/usr/bin/env python3
"""
数据处理器模块
"""

import sys
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import settings
from logger import data_processor_logger as logger
from data_pipeline.collectors.collector import DataCollector
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from .task_manager import TaskManager, DataTask, TaskStatus


class DataProcessor:
    """Data processor for managing scheduled data collection tasks."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.task_manager = TaskManager()
        self.data_collector = DataCollector()
        self.scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)
        self._is_running = False
    
    def start(self):
        """Start the data processor scheduler."""
        try:
            if self._is_running:
                logger.warning("Data processor is already running")
                return
            
            # Add scheduled job for data updates
            self.scheduler.add_job(
                func=self._run_all_tasks,
                trigger=IntervalTrigger(minutes=settings.update_interval_minutes),
                id='data_update_job',
                name='Data Update Job',
                replace_existing=True
            )
            
            self.scheduler.start()
            self._is_running = True
            
            logger.info(f"Data processor started with {settings.update_interval_minutes} minute intervals")
            
        except Exception as e:
            logger.error(f"Error starting data processor: {e}")
            raise
    
    def stop(self):
        """Stop the data processor scheduler."""
        try:
            if not self._is_running:
                logger.warning("Data processor is not running")
                return
            
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Data processor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data processor: {e}")
            raise
    
    def is_running(self) -> bool:
        """Check if the data processor is running."""
        return self._is_running and self.scheduler.running
    
    def add_task(self, symbol: str, timeframe: str, instrument_type: str = "spot") -> int:
        """
        Add a new data collection task.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data collection
            instrument_type: Type of instrument (spot/swap)
            
        Returns:
            Task ID
        """
        try:
            # Validate symbol and timeframe
            if not self.data_collector.client.validate_symbol(symbol):
                raise ValueError(f"Invalid symbol: {symbol}")
            
            if not self.data_collector.client.validate_timeframe(timeframe):
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            task = DataTask(
                exchange="okx",
                symbol=symbol,
                timeframe=timeframe,
                instrument_type=instrument_type,
                status=TaskStatus.ACTIVE.value
            )
            
            task_id = self.task_manager.add_task(task)
            logger.info(f"Added new task {task_id}: {symbol} {timeframe}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error adding task for {symbol} {timeframe}: {e}")
            raise
    
    def remove_task(self, task_id: int):
        """Remove a data collection task."""
        try:
            self.task_manager.delete_task(task_id)
            logger.info(f"Removed task {task_id}")
            
        except Exception as e:
            logger.error(f"Error removing task {task_id}: {e}")
            raise
    
    def pause_task(self, task_id: int):
        """Pause a data collection task."""
        try:
            task = self.task_manager.get_task(task_id)
            if task:
                task.status = TaskStatus.PAUSED.value
                self.task_manager.update_task(task)
                logger.info(f"Paused task {task_id}")
            else:
                logger.warning(f"Task {task_id} not found")
                
        except Exception as e:
            logger.error(f"Error pausing task {task_id}: {e}")
            raise
    
    def resume_task(self, task_id: int):
        """Resume a paused data collection task."""
        try:
            task = self.task_manager.get_task(task_id)
            if task:
                task.status = TaskStatus.ACTIVE.value
                task.error_count = 0  # Reset error count when resuming
                task.error_message = None
                self.task_manager.update_task(task)
                logger.info(f"Resumed task {task_id}")
            else:
                logger.warning(f"Task {task_id} not found")
                
        except Exception as e:
            logger.error(f"Error resuming task {task_id}: {e}")
            raise
    
    def run_task_now(self, task_id: int) -> bool:
        """Run a specific task immediately."""
        try:
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return False
            
            return self._run_single_task(task)
            
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")
            return False
    
    def get_all_tasks(self) -> List[DataTask]:
        """Get all data collection tasks."""
        return self.task_manager.get_all_tasks()
    
    def get_active_tasks(self) -> List[DataTask]:
        """Get all active data collection tasks."""
        return self.task_manager.get_active_tasks()
    
    def get_task_status(self, task_id: int) -> Optional[DataTask]:
        """Get the status of a specific task."""
        return self.task_manager.get_task(task_id)
    
    def _run_all_tasks(self):
        """Run all active data collection tasks."""
        try:
            active_tasks = self.task_manager.get_active_tasks()
            logger.info(f"Running {len(active_tasks)} active tasks")
            
            success_count = 0
            error_count = 0
            
            for task in active_tasks:
                try:
                    if self._run_single_task(task):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error running task {task.id}: {e}")
                    error_count += 1
                
                # Small delay between tasks to avoid overwhelming the API
                time.sleep(1)
            
            logger.info(f"Task run completed: {success_count} successful, {error_count} errors")
            
        except Exception as e:
            logger.error(f"Error running all tasks: {e}")
    
    def _run_single_task(self, task: DataTask) -> bool:
        """
        Run a single data collection task.
        
        Args:
            task: DataTask to run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Running task {task.id}: {task.symbol} {task.timeframe}")
            
            # Update data using the data collector
            success = self.data_collector.update_data(
                symbol=task.symbol,
                timeframe=task.timeframe,
                instrument_type=task.instrument_type
            )
            
            if success:
                self.task_manager.mark_task_success(task.id)
                logger.info(f"Task {task.id} completed successfully")
                return True
            else:
                self.task_manager.mark_task_error(task.id, "Data update failed")
                logger.error(f"Task {task.id} failed")
                return False
                
        except Exception as e:
            error_message = str(e)
            self.task_manager.mark_task_error(task.id, error_message)
            logger.error(f"Error running task {task.id}: {error_message}")
            return False
    
    def get_scheduler_info(self) -> dict:
        """Get information about the scheduler and its jobs."""
        try:
            if not self.scheduler:
                return {"status": "not_initialized"}
            
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
            
            return {
                "status": "running" if self.is_running() else "stopped",
                "timezone": str(self.scheduler.timezone),
                "jobs": jobs,
                "update_interval_minutes": settings.update_interval_minutes
            }
            
        except Exception as e:
            logger.error(f"Error getting scheduler info: {e}")
            return {"status": "error", "error": str(e)}